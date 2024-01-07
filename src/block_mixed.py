import torch
from torch import nn
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from utils import get_rw_adj, add_remaining_self_loops


class MixedODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, device, t=torch.tensor([0, 1]), gamma=0.):
    super(MixedODEblock, self).__init__(odefunc, regularization_fns, opt, device, t)

    self.device = device
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.gamma = nn.Parameter(gamma * torch.ones(1))
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device).to(device)

  def reset_graph_data(self, data, dtype):
    self.num_nodes = data.num_nodes
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=self.opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=dtype)
    if self.opt['self_loop_weight'] > 0:
      edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                         fill_value=self.opt['self_loop_weight'], num_nodes=data.num_nodes)
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

  
  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def get_mixed_attention(self, x):
    gamma = torch.sigmoid(self.gamma)
    attention = self.get_attention_weights(x)
    mixed_attention = attention.mean(dim=2) * (1 - gamma) + self.odefunc.edge_weight * gamma
    return mixed_attention

  def forward(self, x, graph_data):
    t = self.t.type_as(x)
    self.reset_graph_data(graph_data, x.dtype)
    self.odefunc.attention_weights = self.get_mixed_attention(x)
    integrator = self.train_integrator if self.training else self.test_integrator
    if self.opt["adjoint"] and self.training:
      z = integrator(
        self.odefunc, x, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)[1]
    else:
      z = integrator(
        self.odefunc, x, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)[1]

    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
