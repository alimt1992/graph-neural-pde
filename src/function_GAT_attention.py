import torch
from torch import nn
import numpy as np
from torch_geometric.utils import softmax
import torch_sparse
from data import get_dataset
from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt, device):
    super(ODEFuncAtt, self).__init__(opt, device)

    self.device = device
    self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features, opt,
                                                     device).to(device)
    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

  def multiply_attention(self, x, attention, wx):
    with (torch.device(self.device)):
      if self.opt['mix_features']:
        index0 = torch.arange(self.edge_index.shape[0])[:, None, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index1 = self.edge_index[:,0][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index2 = self.edge_index[:,1][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index3 = torch.arange(self.opt['heads'])[None, None, :].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        indices = torch.stack([index0, index1, index2, index3] , dim=0)
        sparse_att = torch.sparse_coo_tensor(indices, attention.flatten(), [wx.shape[0], wx.shape[1], wx.shape[1], self.opt['heads']],
                                             requires_grad=True).to(self.device)
        wx = torch.mean(torch.matmul(sparse_att.permute(0,3,1,2).to_dense(), wx.unsqueeze(1)), dim=1)
        ax = torch.matmul(wx, self.multihead_att_layer.Wout)
      else:
        index0 = torch.arange(self.edge_index.shape[0])[:, None, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index1 = self.edge_index[:,0][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index2 = self.edge_index[:,1][:, :, None].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        index3 = torch.arange(self.opt['heads'])[None, None, :].expand(self.edge_index.shape[0], self.edge_index.shape[2], self.opt['heads']).flatten()
        indices = torch.stack([index0, index1, index2, index3] , dim=0)
        sparse_att = torch.sparse_coo_tensor(indices, attention.flatten(), [x.shape[0], x.shape[1], x.shape[1], self.opt['heads']],
                                             requires_grad=True).to(self.device)
        ax = torch.mean(torch.matmul(sparse_att.permute(0,3,1,2).to_dense(), x.unsqueeze(1)), dim=1)
    return ax

  def forward(self, t, x, y=None):  # t is needed when called by the integrator

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1

    attention, wx = self.multihead_att_layer(x, self.edge_index, y)
    ax = self.multiply_attention(x, attention, wx)
    # todo would be nice if this was more efficient

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


class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = opt['heads']

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414)

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(1, 2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

    #if there is other modality
    if self.opt['multi_modal']:
      self.Q2 = nn.Linear(in_features, in_features)
      self.init_weights(self.Q2)
      self.V2 = nn.Linear(opt['second_modality_dim'], in_features)
      self.init_weights(self.V2)
      self.K2 = nn.Linear(opt['second_modality_dim'], in_features)
      self.init_weights(self.K2)

  def forward(self, x, edge, y=None):

    if self.opt['multi_modal']:
        dk = self.in_features
        x = torch.matmul(torch.nn.softmax(torch.matmul(self.Q2(x), self.K2(y).transpose(-2, -1) / np.sqrt(dk))), self.V2(y))
    
    wx = torch.matmul(x, self.W)  # h: N x out
    h = wx.view(self.opt['batch_size'], -1, self.h, self.d_k)
    h = h.transpose(2, 3)

    # Self-attention on the nodes - Shared attention mechanism
    index0 = torch.arange(h.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    index2 = torch.arange(h.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    index3 = torch.arange(h.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    edge_h = torch.cat((h[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(3), index2, index3],
                        h[index0, edge[:, 1, :].unsqueeze(2).unsqueeze(3), index2, index3]),
                        dim=2).transpose(1, 2).to(self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=1)).to(self.device)
    attention = torch.stack([softmax(edge_e[i], edge[i,self.opt['attention_norm_idx'],:]) for i in range(edge_e.shape[0])], dim=0)###
    return attention, wx

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10, 'attention_norm_idx': 0,
         'add_source':False, 'alpha_dim': 'sc', 'beta_dim': 'vc', 'max_nfe':1000, 'mix_features': False}
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncAtt(dataset.data.num_features, 6, opt, device)
  out = func(t, dataset.data.x)
