import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from utils import get_rw_adj, gcn_norm_fill_val, add_remaining_self_loops
from utils import Meter
from regularized_ODE_function import RegularizedODEfunc
import regularized_ODE_function as reg_lib
import six
from functools import partial


REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs


class ODEblock(nn.Module):
  def __init__(self, odefunc, regularization_fns, opt, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t
    
    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, device)
    self.ode_last_call_time = [0 for i in range(self.opt['ode_order'])]
    #storing function time on last call in order to avoid recomputation
    self.ode_last_integrals = [0 for i in range(self.opt['ode_order'])]
    #storing function value on last call in order to avoid recomputation
    #need to be set befor each call as the initial value problem needs them
    
    self.nreg = len(regularization_fns)
    if self.opt['ode_order'] == 1:
      self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)
    else:
      func = partial(self, graph_data=None, y=None, ode_order=self.opt['ode_order']-1)
      #recursive function calls for higher order pdes
      self.reg_odefunc = RegularizedODEfunc(func, regularization_fns)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    self.reg_odefunc.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    if self.opt['adjoint']:
      self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
      self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9
  
  def reset_graph_data(self, data, dtype, y=None):
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
    self.odefunc.y = y
    self.data_edge_index = edge_index.to(self.device)
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight


  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def __repr__(self):
    return self.__class__.__name__


class BaseGNN(MessagePassing):
  def __init__(self, opt, num_classes, num_features, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = num_classes
    self.num_features = num_features
    self.device = device
    self.fm = Meter()
    self.bm = Meter()

    if opt['beltrami']:
      self.mx = nn.Linear(self.num_features, opt['feat_hidden_dim'])
      self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
      opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']
    else:
      self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])

    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    if opt['use_labels']:
      # todo - fastest way to propagate this everywhere, but error prone - refactor later
      opt['hidden_dim'] = opt['hidden_dim'] + num_classes
      self.hidden_dim = opt['hidden_dim']
    else:
      self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.m2 = nn.Linear(opt['hidden_dim'], num_classes)
    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__
