import torch
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from torch_scatter import scatter

class HardAttODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(HardAttODEblock, self).__init__(odefunc, regularization_fns, opt, device, t)
    assert opt['att_samp_pct'] > 0 and opt['att_samp_pct'] <= 1, "attention sampling threshold must be in (0,1]"
    self.device = device
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    if opt['function'] not in {'GAT', 'transformer'}:
      self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,
                                                          device, edge_weights=self.odefunc.edge_weight).to(device)

  def get_attention_weights(self, x):
    if self.opt['function'] not in {'GAT', 'transformer'}:
      attention, values = self.multihead_att_layer(x, self.data_edge_index)
    else:
      attention, values = self.odefunc.multihead_att_layer(x, self.data_edge_index)
    return attention

  def renormalise_attention(self, attention):
    index = self.odefunc.edge_index[self.opt['attention_norm_idx']]
    att_sums = scatter(attention, index, dim=1, dim_size=self.num_nodes, reduce='sum')[index]
    return attention / (att_sums + 1e-16)

  def forward(self, x, graph_data, y=None):
    t = self.t.type_as(x)
    self.reset_graph_data(graph_data, x.dtype, y)
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
          delta = torch.linalg.norm(src_features-dst_features, dim=1)
          mean_att = mean_att * delta
        threshold = torch.quantile(mean_att, 1-self.opt['att_samp_pct'])
        mask = mean_att > threshold
        self.odefunc.edge_index = self.data_edge_index[:, :, mask.T]
        sampled_attention_weights = self.renormalise_attention(mean_att[mask])
        print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[2], self.data_edge_index.shape[2]))
        self.odefunc.attention_weights = sampled_attention_weights
    else:
      self.odefunc.edge_index = self.data_edge_index
      self.odefunc.attention_weights = attention_weights.mean(dim=2, keepdim=False)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator

    reg_states = tuple(torch.zeros(x.size(0), x.size(1)).to(x) for i in range(self.nreg))

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size']},
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states
    else:
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
