from base_classes import ODEblock
import torch
from functools import partial


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
  
  def forward(self, t, x, graph_data, ode_order=1, y=None):
    t_range = self.t.type_as(x)

    if ode_order==self.opt['ode_order']:
      self.reset_graph_data(graph_data, x.dtype, y)

      reg_states = tuple( torch.zeros(x.size(0), x.size(1)).to(x) for i in range(self.nreg) )

      func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc if self.opt['order']==1 \
        else partial(self, graph_data=None, y=None, ode_order=self.opt['ode_order']-1)
      state = (x,) + reg_states if self.training and self.nreg > 0 else x
    elif ode_order!=1:
      func = partial(self, graph_data=None, y=None, ode_order=ode_order-1)
      #state = x
    else:
      func = self.odefunc
      #state = x
    
    integrator = self.train_integrator if self.training else self.test_integrator

    if ode_order!=self.opt['ode_order']:
      t_range[0] = self.ode_last_call_time[ode_order-1]
      t_range[1] = t
      state = self.ode_last_integrals[ode_order-1]
    
    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t_range,
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
        func, state, t_range,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0 and ode_order==self.opt['ode_order']:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      self.ode_last_call_time = [0 for i in range(self.opt['ode_order'])]
      self.ode_last_integrals = [0 for i in range(self.opt['ode_order'])]
      return z, reg_states
    elif ode_order==self.opt['ode_order']:
      z = state_dt[1]
      self.ode_last_call_time = [0 for i in range(self.opt['ode_order'])]
      self.ode_last_integrals = [0 for i in range(self.opt['ode_order'])]
      return z
    else:
      z = state_dt[1]
      self.ode_last_integrals[ode_order-1] = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
