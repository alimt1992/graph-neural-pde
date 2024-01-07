from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val, add_remaining_self_loops
import torch_sparse
from torch_geometric.utils import get_laplacian
import numpy as np

class ConstantODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, device, t)

    self.device = device
    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()

  def reset_graph_data(self, data, dtype):
    self.num_nodes = data.num_nodes
    if self.opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                                  fill_value=self.opt['self_loop_weight'],
                                                  num_nodes=data.num_nodes,
                                                  dtype=dtype)
    if self.opt['self_loop_weight'] > 0:
      edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                         fill_value=self.opt['self_loop_weight'], num_nodes=data.num_nodes)
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

  def add_random_edges(self):
    #todo check if theres a pygeometric function for this

    # M = self.opt["M_nodes"]
    M = int(self.num_nodes * (1/(1 - (1 - self.opt['att_samp_pct'])) - 1))
    n = self.num_nodes

    with torch.no_grad():
      new_edges = np.random.choice(self.num_nodes, size=(self.opt['batch_size'],2,M), replace=True, p=None)
      new_edges = torch.tensor(new_edges)
      cat = torch.cat([self.data_edge_index, new_edges],dim=2)
      index0 = torch.arange(cat.shape[0])[:, None].expand(cat.shape[0], cat.shape[2]).flatten()
      index1 = cat[:,0].flatten()
      index2 = cat[:,1].flatten()
      indices = torch.stack([index0, index1, index2] , dim=0)
      edge_mat = torch.sparse_coo_tensor(indices, torch.ones(cat.shape[0], cat.shape(2)), [cat.shape[0], n, n],
                                         requires_grad=True).to(self.device).to_dense()
      nonzero_mask = torch.zeros_like(edge_mat)
      nonzero_mask[edge_mat != 0] = 1
      nonzero_mask = nonzero_mask.reshape(-1, n*n).unsqueeze(2)
      new_edges = torch.cartesian_prod(torch.arange(n), torch.arange(n))[None,:,:].expand(cat.shape[0], n*n, 2)
      new_edges = new_edges * nonzero_mask
      sorted, _ = torch.sort(new_edges, dim=1, descending=True)
      no_repeats = torch.unique_consecutive(sorted, dim=1).transpose(1,2)
      self.data_edge_index = no_repeats

  def add_khop_edges(self, k):
    n = self.num_nodes
    # do k_hop
    for i in range(k):
      index0 = torch.arange(self.odefunc.edge_index.shape[0])[:, None].expand(self.odefunc.edge_index.shape[0], self.odefunc.edge_index.shape[2]).flatten()
      index1 = self.odefunc.edge_index[:,0].flatten()
      index2 = self.odefunc.edge_index[:,1].flatten()
      indices = torch.stack([index0, index1, index2] , dim=0)
      weight_mat = torch.sparse_coo_tensor(indices, self.odefunc.edge_weight.flatten(), [self.odefunc.edge_index.shape[0], n, n],
                                                requires_grad=True).to(self.device).to_dense()
      new_weights = torch.matmul(weight_mat, weight_mat)
      edge_weights = 0.5 * weight_mat + 0.5 * new_weights
      zero_mask = 1-torch.eye(n)[None,:,:].expand(self.odefunc.edge_index.shape[0], n, n)
      edge_weights = edge_weights * zero_mask
      
      nonzero_mask = torch.zeros_like(edge_weights)
      nonzero_mask[new_weights != 0] = 1
      nonzero_mask = nonzero_mask.reshape(-1, n*n).unsqueeze(2)
      new_edges = torch.cartesian_prod(torch.arange(n), torch.arange(n))[None,:,:].expand(self.odefunc.edge_index.shape[0], n*n, 2)
      new_edges = new_edges * nonzero_mask
      sorted, _ = torch.sort(new_edges, dim=1, descending=True)
      new_edges = torch.unique_consecutive(sorted, dim=1).transpose(1,2)
      
      index0 = torch.arange(new_edges.shape[0])[:, None].expand(new_edges.shape[0], new_edges.shape[2])
      index1 = new_edges[:, 0, :]
      index2 = new_edges[:, 1, :]
      self.edge_weight = edge_weights[index0, index1, index2]
      self.odefunc.edge_weight = self.edge_weight
      self.data_edge_index = new_edges
    # threshold
    # normalise

  # self.odefunc.edge_index, self.odefunc.edge_weight =
  # get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  # num_nodes = maybe_num_nodes(edge_index, num_nodes)


  def forward(self, x, graph_data):
    t = self.t.type_as(x)

    self.reset_graph_data(graph_data, x.dtype)

    if self.training:
      if self.opt['new_edges'] == 'random':
        self.add_random_edges()
      elif self.opt['new_edges'] == 'k_hop':
        self.add_khop_edges(k=2)
      elif self.opt['new_edges'] == 'random_walk' and self.odefunc.attention_weights is not None:
        self.add_rw_edges()



    attention_weights = self.get_attention_weights(x)
    # create attention mask
    if self.training:
      with torch.no_grad():
        mean_att = attention_weights.mean(dim=2, keepdim=False)
        if self.opt['use_flux']:
          index0 = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)
          index2 = torch.arange(x.shape[2]).unsqueeze(0).unsqueeze(0)
          src_features = x[index0, self.data_edge_index[:, 0, :].unsqueeze(1), index2]
          dst_features = x[index0, self.data_edge_index[:, 1, :].unsqueeze(1), index2]
          delta = torch.linalg.norm(src_features - dst_features, dim=2)
          mean_att = mean_att * delta
        threshold = torch.quantile(mean_att, 1-self.opt['att_samp_pct'])
        mask = mean_att > threshold
        index0 = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)
        index1 = torch.arange(2).unsqueeze(0).unsqueeze(2)
        index2 = mask.unsqueeze(1)
        # mask = mask.transpose(0,1)
        self.odefunc.edge_index = self.data_edge_index[index0, index1, index2]
        sampled_attention_weights = self.renormalise_attention(mean_att[mask])
        print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[2], self.data_edge_index.shape[2]))
        self.odefunc.edge_weight = sampled_attention_weights
        self.odefunc.attention_weights = sampled_attention_weights
    else:
      self.odefunc.edge_index = self.data_edge_index
      self.odefunc.attention_weights = attention_weights.mean(dim=2, keepdim=False)
      self.odefunc.edge_weight = self.odefunc.attention_weights
    
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights




    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple( torch.zeros(x.size(0), x.size(1)).to(x) for i in range(self.nreg) )

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      return z, reg_states
    else: 
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
