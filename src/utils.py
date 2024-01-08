"""
utility functions
"""
import os
import torch
import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils.num_nodes import maybe_num_nodes

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MaxNFEException(Exception): pass


def add_remaining_self_loops(edge_index, edge_attr, fill_value, num_nodes):
  n = num_nodes

  index0 = torch.arange(edge_index.shape[0])[:, None].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  index1 = edge_index[:,0].flatten()
  index2 = edge_index[:,1].flatten()
  indices = torch.stack([index0, index1, index2] , dim=0)
  weight_mat = torch.sparse_coo_tensor(indices, edge_attr.flatten(), [edge_index.shape[0], n, n],
                                            requires_grad=True).to_dense()
  edge_weights = weight_mat
  loop_weights = torch.eye(n)[None,:,:].expand(edge_index.shape[0], n, n) * fill_value
  edge_weights = torch.sum(torch.cat((edge_weights[:,:,:,None], loop_weights[:,:,:,None]), dim=3), dim=3)
  nonzero_mask = torch.zeros_like(edge_weights)
  nonzero_mask[edge_weights != 0] = 1
  nonzero_mask = nonzero_mask.reshape(-1, n*n).unsqueeze(2)
  new_edges = torch.cartesian_prod(torch.arange(n), torch.arange(n))[None,:,:].expand(edge_index.shape[0], n*n, 2)
  new_edges = new_edges * nonzero_mask
  sorted, _ = torch.sort(new_edges, dim=1, descending=True)
  new_edges = torch.unique_consecutive(sorted, dim=1).transpose(1,2).long()
  
  index0 = torch.arange(new_edges.shape[0])[:, None].expand(new_edges.shape[0], new_edges.shape[2])
  index1 = new_edges[:, 0, :]
  index2 = new_edges[:, 1, :]
  edge_index = new_edges
  edge_attr = edge_weights[index0, index1, index2]

  return edge_index, edge_attr

def remove_self_loops(edge_index, edge_attr):
  pass

def to_dense_adj(edge_index, edge_attr=None):
  n = maybe_num_nodes(edge_index, None)
  index0 = torch.arange(edge_index.shape[0])[:, None].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  index1 = edge_index[:,0].flatten()
  index2 = edge_index[:,1].flatten()
  indices = torch.stack([index0, index1, index2] , dim=0)
  adj_mat = torch.sparse_coo_tensor(indices, edge_attr.flatten(), [edge_index.shape[0], n, n],
                                            requires_grad=True).to_dense()
  return adj_mat


def softmax(src, index, num_nodes=None):
  num_nodes = maybe_num_nodes(index, num_nodes)

  index0 = torch.arange(src.shape[0]).unsqueeze(1).unsqueeze(2)
  index1 = index.unsqueeze(2)
  index2 = torch.arange(src.shape[2]).unsqueeze(0).unsqueeze(0)
  out = src - scatter_max(src, index, dim=1, dim_size=num_nodes)[0][index0, index1, index2]
  out = out.exp()
  out = out / (
      scatter_add(out, index, dim=1, dim_size=num_nodes)[index0, index1, index2] + 1e-16)

  return out

def squareplus(src, index, num_nodes):
  num_nodes = maybe_num_nodes(index, num_nodes)
  
  out = src - src.amax(dim=[1,2], keepdim=True)
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  index0 = torch.arange(src.shape[0]).unsqueeze(1).unsqueeze(2)
  index1 = index.unsqueeze(2)
  index2 = torch.arange(src.shape[2]).unsqueeze(0).unsqueeze(0)
  out_sum = scatter_add(out, index, dim=1, dim_size=num_nodes)[index0, index1, index2]

  return out / (out_sum + 1e-16)


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[:, 0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[:, 1:1 + state_size]
    adj_y = aug_state[:, 1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(0), edge_index.size(2),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[:, 0], edge_index[:, 1]
  deg = scatter_add(edge_weight, col, dim=1, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(edge_index, edge_weight, device=None):
  n = maybe_num_nodes(edge_index, n)
  index0 = torch.arange(edge_index.shape[0])[:, None].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  index1 = edge_index[:,0].flatten()
  index2 = edge_index[:,1].flatten()
  indices = torch.stack([index0, index1, index2] , dim=0)
  weight_mat = torch.sparse_coo_tensor(indices, edge_weight.flatten(), [edge_index.shape[0], n, n],
                                            requires_grad=True, device=device)
  return weight_mat


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm_fill_val(  # yapf: disable
    data.edge_index, data.edge_attr, opt['self_loop_weight'] > 0,
    data.num_nodes, dtype=data.x.dtype)
  return coo2tensor(edge_index, edge_weight)


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(0), edge_index.size(2),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[:, 0], edge_index[:, 1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=1, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(batch_size, num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((batch_size, 2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[:, 0, idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[:, 1, idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)[None, :].expand(batch_size, num_nodes)
  return edge_index


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
