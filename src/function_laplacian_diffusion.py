import torch
from torch import nn
import numpy as np
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

    #if there is other modality
    if self.opt['multi_modal']:
      self.Q2 = nn.Linear(in_features, in_features)
      self.init_weights(self.Q2)
      self.V2 = nn.Linear(opt['second_modality_dim'], in_features)
      self.init_weights(self.V2)
      self.K2 = nn.Linear(opt['second_modality_dim'], in_features)
      self.init_weights(self.K2)

  def sparse_multiply(self, x):
    index0 = torch.arange(self.edge_index.shape[0])[:, None].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
    index1 = self.edge[:,0].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
    index2 = self.edge[:,1].expand(self.edge_index.shape[0], self.edge_index.shape[2]).flatten()
    indices = torch.stack([index0, index1, index2] , dim=0)
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=2)
      # ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[1], x.shape[1], x)
      sparse_att = torch.sparse_coo_tensor(indices, mean_attention.flatten(), [x.shape[0], x.shape[1], x.shape[1]],
                                           requires_grad=True).to(self.device)
      ax = torch.matmul(sparse_att.to_dense(), x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      # ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[1], x.shape[1], x)
      sparse_att = torch.sparse_coo_tensor(indices, self.attention_weights.flatten(), [x.shape[0], x.shape[1], x.shape[1]],
                                           requires_grad=True).to(self.device)
      ax = torch.matmul(sparse_att.to_dense(), x)
    else:  # adj is a torch sparse matrix
      # ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[1], x.shape[1], x)
      sparse_att = torch.sparse_coo_tensor(indices, self.edge_weight.flatten(), [x.shape[0], x.shape[1], x.shape[1]],
                                           requires_grad=True).to(self.device)
      ax = torch.matmul(sparse_att.to_dense(), x)
    return ax

  def forward(self, t, x, y=None):  # the t param is needed by the ODE solver.
    if self.opt['multi_modal']:
        dk = self.in_features
        x = torch.matmul(torch.nn.softmax(torch.matmul(self.Q2(x), self.K2(y).transpose(-2, -1) / np.sqrt(dk))), self.V2(y))
    
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
