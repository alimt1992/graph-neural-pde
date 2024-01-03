import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc


class ODEFuncTransformerAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncTransformerAtt, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                          device, edge_weights=self.edge_weight).to(device)

  def multiply_attention(self, x, attention, v=None):
    # todo would be nice if this was more efficient
    if self.opt['mix_features']:
      # vx = torch.mean(torch.stack(
      #   [torch_sparse.spmm(self.edge_index, attention[:, :, idx], v.shape[1], v.shape[1], v[:, :, :, idx]) for idx in
      #    range(self.opt['heads'])], dim=1),
      #   dim=1)
      index0 = torch.arange(self.edge_index.shape[0])[:, None, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
      index1 = self.edge[:,0][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
      index2 = self.edge[:,1][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
      index3 = torch.arange(self.opt['heads'])[None, None, :].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
      indices = torch.stack([index0, index1, index2, index3] , dim=0)
      sparse_att = torch.sparse_coo_tensor(indices, attention.flatten(), [v.shape[0], v.shape[1], v.shape[1], self.opt['heads']],
                                           requires_grad=True).to(self.device)
      vx = torch.mean(torch.matmul(sparse_att.permute(0,3,1,2).to_dense(), v.permute(0,3,1,2)), dim=1)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=2)
      # ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[1], x.shape[1], x)
      index0 = torch.arange(self.edge_index.shape[0])[:, None].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
      index1 = self.edge[:,0].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
      index2 = self.edge[:,1].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
      indices = torch.stack([index0, index1, index2] , dim=0)
      sparse_att = torch.sparse_coo_tensor(indices, mean_attention.flatten(), [x.shape[0], x.shape[1], x.shape[1]],
                                           requires_grad=True).to(self.device)
      ax = torch.matmul(sparse_att.to_dense(), x)
    return ax

  def forward(self, t, x, y=None):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    attention, values = self.multihead_att_layer(x, self.edge_index, y)
    ax = self.multiply_attention(x, attention, values)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      self.output_var_x = nn.Parameter(torch.ones(1))
      self.lengthscale_x = nn.Parameter(torch.ones(1))
      self.output_var_p = nn.Parameter(torch.ones(1))
      self.lengthscale_p = nn.Parameter(torch.ones(1))
      self.Qx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qx)
      self.Vx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vx)
      self.Kx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kx)

      self.Qp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qp)
      self.Vp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vp)
      self.Kp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kp)

      #if there is other modality
      if self.opt['multi_modal']:
        self.Q2 = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], opt['hidden_dim']-opt['pos_enc_hidden_dim'])
        self.init_weights(self.Q2)
        self.V2 = nn.Linear(opt['second_modality_dim'], opt['hidden_dim']-opt['pos_enc_hidden_dim'])
        self.init_weights(self.V2)
        self.K2 = nn.Linear(opt['second_modality_dim'], opt['hidden_dim']-opt['pos_enc_hidden_dim'])
        self.init_weights(self.K2)

        self.Q2p = nn.Linear(opt['pos_enc_hidden_dim'], opt['pos_enc_hidden_dim'])
        self.init_weights(self.Q2p)
        self.V2p = nn.Linear(opt['second_modality_dim'], opt['pos_enc_hidden_dim'])
        self.init_weights(self.V2p)
        self.K2p = nn.Linear(opt['second_modality_dim'], opt['pos_enc_hidden_dim'])
        self.init_weights(self.K2p)

    else:
      if self.opt['attention_type'] == "exp_kernel":
        self.output_var = nn.Parameter(torch.ones(1))
        self.lengthscale = nn.Parameter(torch.ones(1))

      self.Q = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.Q)

      self.V = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.V)

      self.K = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.K)

      #if there is other modality
      if self.opt['multi_modal']:
        self.Q2 = nn.Linear(in_features, in_features)
        self.init_weights(self.Q2)
        self.V2 = nn.Linear(opt['second_modality_dim'], in_features)
        self.init_weights(self.V2)
        self.K2 = nn.Linear(opt['second_modality_dim'], in_features)
        self.init_weights(self.K2)
    
    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge, y=None):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      x = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)

      if self.opt['multi_modal']:
        dk = self.opt['hidden_dim']-self.opt['pos_enc_hidden_dim']
        x = torch.matmul(torch.nn.softmax(torch.matmul(self.Q2(x), self.K2(y).transpose(-2, -1) / np.sqrt(dk)), dim=-1), self.V2(y))
        dk = self.opt['pos_enc_hidden_dim']
        p = torch.matmul(torch.nn.softmax(torch.matmul(self.Q2p(p), self.K2p(y).transpose(-2, -1) / np.sqrt(dk)), dim=-1), self.V2p(y))
      
      qx = self.Qx(x)
      kx = self.Kx(x)
      vx = self.Vx(x)
      # perform linear operation and split into h heads
      kx = kx.view(self.opt['batch_size'], -1, self.h, self.d_k)
      qx = qx.view(self.opt['batch_size'], -1, self.h, self.d_k)
      vx = vx.view(self.opt['batch_size'], -1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kx = kx.transpose(2, 3)
      qx = qx.transpose(2, 3)
      vx = vx.transpose(2, 3)

      index0 = torch.arange(qx.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
      index2 = torch.arange(qx.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
      index3 = torch.arange(qx.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
      src_x = qx[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(2), index2, index3]
      dst_x = kx[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(2), index2, index3]

      qp = self.Qp(p)
      kp = self.Kp(p)
      vp = self.Vp(p)
      # perform linear operation and split into h heads
      kp = kp.view(self.opt['batch_size'], -1, self.h, self.d_k)
      qp = qp.view(self.opt['batch_size'], -1, self.h, self.d_k)
      vp = vp.view(self.opt['batch_size'], -1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kp = kp.transpose(2, 3)
      qp = qp.transpose(2, 3)
      vp = vp.transpose(2, 3)

      index0 = torch.arange(qp.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
      index2 = torch.arange(qp.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
      index3 = torch.arange(qp.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
      src_p = qp[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(2), index2, index3]
      dst_p = kp[index0, edge[:, 1, :].unsqueeze(2).unsqueeze(2), index2, index3]

      prods = self.output_var_x ** 2 * torch.exp(
        -torch.sum((src_x - dst_x) ** 2, dim=2) / (2 * self.lengthscale_x ** 2)) \
              * self.output_var_p ** 2 * torch.exp(
        -torch.sum((src_p - dst_p) ** 2, dim=2) / (2 * self.lengthscale_p ** 2))

      v = None

    else:

      if self.opt['multi_modal']:
        dk = self.in_features
        x = torch.matmul(torch.nn.softmax(torch.matmul(self.Q2(x), self.K2(y).transpose(-2, -1) / np.sqrt(dk))), self.V2(y))
      
      q = self.Q(x)
      k = self.K(x)
      v = self.V(x)

      # perform linear operation and split into h heads

      k = k.view(self.opt['batch_size'], -1, self.h, self.d_k)
      q = q.view(self.opt['batch_size'], -1, self.h, self.d_k)
      v = v.view(self.opt['batch_size'], -1, self.h, self.d_k)

      # transpose to get dimensions [n_nodes, attention_dim, n_heads]

      k = k.transpose(2, 3)
      q = q.transpose(2, 3)
      v = v.transpose(2, 3)

      index0 = torch.arange(q.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
      index2 = torch.arange(q.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
      index3 = torch.arange(q.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
      src = q[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(2), index2, index3]
      dst_k = k[index0, edge[:, 1, :].unsqueeze(2).unsqueeze(2), index2, index3]

    if not self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      prods = self.output_var ** 2 * torch.exp(-(torch.sum((src - dst_k) ** 2, dim=2) / (2 * self.lengthscale ** 2)))
    elif self.opt['attention_type'] == "scaled_dot":
      prods = torch.sum(torch.matmul(src.permute(0,3,1,2), dst_k.permute(0,3,2,1) / np.sqrt(self.d_k)), dim=3).permute(0,2,1)
    elif self.opt['attention_type'] == "cosine_sim":
      cos = torch.nn.CosineSimilarity(dim=2, eps=1e-5)
      prods = cos(src, dst_k)
    elif self.opt['attention_type'] == "pearson":
      src_mu = torch.mean(src, dim=2, keepdim=True)
      dst_mu = torch.mean(dst_k, dim=2, keepdim=True)
      src = src - src_mu
      dst_k = dst_k - dst_mu
      cos = torch.nn.CosineSimilarity(dim=2, eps=1e-5)
      prods = cos(src, dst_k)

    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=2)
    if self.opt['square_plus']:
      attention = torch.stack([squareplus(prods[i], edge[i,self.opt['attention_norm_idx'],:]) for i in range(prods.shape[0])], dim=0)###
    else:
      attention = torch.stack([softmax(prods[i], edge[i,self.opt['attention_norm_idx'],:]) for i in range(prods.shape[0])], dim=0)###
    return attention, (v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
         'attention_norm_idx': 0, 'add_source': False,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncTransformerAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
