import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
from data import get_dataset
from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncAtt, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features, opt,
                                                     device).to(device)
    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

  def multiply_attention(self, x, attention, wx):
    if self.opt['mix_features']:
      wx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], wx.shape[1], wx.shape[1], wx) for idx in
         range(self.opt['heads'])], dim=1),
        dim=1)
      ax = torch.mm(wx, self.multihead_att_layer.Wout)
    else:
      ax = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], x.shape[1], x.shape[1], x) for idx in
         range(self.opt['heads'])], dim=1),
        dim=1)
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
        x = torch.mm(torch.nn.softmax(torch.mm(self.Q2(x), self.K2(y).t)), self.V2(y))
    
    wx = torch.mm(x, self.W)  # h: N x out
    h = wx.view(self.opt['batch_size'], -1, self.h, self.d_k)
    h = h.transpose(2, 3)

    # Self-attention on the nodes - Shared attention mechanism
    index0 = torch.arange(h.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    index2 = torch.arange(h.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
    index3 = torch.arange(h.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    edge_h = torch.cat((h[index0, edge[:, 0, :].unsqueeze(2).unsqueeze(2), index2, index3], h[:, edge[:, 1, :], index2, index3]), dim=1).transpose(1, 2).to(
      self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=1)).to(self.device)
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention, wx

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10, 'attention_norm_idx': 0,
         'add_source':False, 'alpha_dim': 'sc', 'beta_dim': 'vc', 'max_nfe':1000, 'mix_features': False}
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
