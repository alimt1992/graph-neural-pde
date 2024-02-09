import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from GNN import GNN
import time
from data_multi import get_dataset


def get_cora_opt(opt):
  opt['dataset'] = 'Cora'
  opt['data'] = 'Planetoid'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'rmsprop'
  opt['lr'] = 0.0047
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  opt['alpha'] = 0.918
  opt['time'] = 12.1
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['epoch'] = 31
  opt['augment'] = True
  opt['attention_dropout'] = 0
  opt['adjoint'] = False
  opt['ode'] = 'ode'
  return opt


def get_computers_opt(opt):
  opt['dataset'] = 'Computers'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'adam'
  opt['lr'] = 0.01
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  opt['alpha'] = 0.918
  opt['epoch'] = 400
  opt['time'] = 12.1
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['epoch'] = 50
  opt['attention_dropout'] = 0
  opt['ode'] = 'ode'
  return opt

def get_clevr_opt(opt):
  opt['dataset'] = 'Computers'
  opt['hidden_dim'] = 16
  opt['input_dropout'] = 0.5
  opt['dropout'] = 0
  opt['optimizer'] = 'adam'
  opt['lr'] = 0.01
  opt['decay'] = 5e-4
  opt['self_loop_weight'] = 0.555
  opt['alpha'] = 0.918
  opt['epoch'] = 400
  opt['time'] = 12.1
  opt['num_feature'] = 1433
  opt['num_class'] = 7
  opt['num_nodes'] = 2708
  opt['epoch'] = 50
  opt['attention_dropout'] = 0
  opt['ode'] = 'ode'
  return opt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, dataset):
  model.train()
  loader = DataLoader(dataset, batch_size=model.opt['batch_size'], shuffle=True)
  total_correct = 0

  for batch_idx, batch in enumerate(loader):
    optimizer.zero_grad()
    start_time = time.time()

    if batch_idx > model.opt['train_size']//model.opt['batch_size']: # only do this for 1st batch/epoch
      break

    out = model(batch['modality_data'], batch['modality_graphs'], batch['additional_data'])

    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out, batch['labels'])  #squeeze now needed

    pred = out.max(1)[1]
    total_correct += pred.eq(batch['labels']).sum().item()

    
    if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
      reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
      regularization_coeffs = model.regularization_coeffs

      reg_loss = sum(
        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
      )
      loss = loss + reg_loss

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()
    if batch_idx % 10000 == 0:
      print("Batch Index {}, number of function evals {} in time {}".format(batch_idx, model.fm.sum, time.time() - start_time))
    
  accs = total_correct / model.opt['train_size']

  return accs, loss.item()


@torch.no_grad()
def test(model, dataset):
  model.eval()
  batch_size = model.opt['batch_size']
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  total_correct = 0
  
  for batch_idx, batch in enumerate(loader):
    if batch_idx > model.opt['val_size']//model.opt['batch_size']: # only do this for 1st batch/epoch
      break
    logits= model(batch['modality_data'], batch['modality_graphs'], batch['additional_data'])
    pred = logits.max(1)[1]
    total_correct += pred.eq(batch['labels']).sum().item()
  accs = total_correct / model.opt['val_size']
  
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


def main(opt, run_count):
  # Load dataset and create model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_dataset, val_dataset = get_dataset(opt, '..', device)
  model= GNN(opt, opt['num_class'], opt['num_features'], device).to(device)
  print(opt)

  # Todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)

  # Training/test loop
  results = {
    'time':[],
    'loss':[],
    'forward_nfe':[],
    'backward_nfe':[],
    'train_acc':[],
    'test_acc':[],
    'val_acc':[],
    'best_epoch':0,
    'best_val_acc':0.,
  }
  runtimes = []
  losses = []
  
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = train_acc = best_epoch = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()

    train_acc, loss = train(model, optimizer, train_dataset)
    val_acc = test(model, val_dataset)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_epoch = epoch

    #if epoch % 10 == 0:
    results['time'].append(time.time() - start_time)
    results['loss'].append(loss)
    results['forward_nfe'].append(model.fm.sum)
    results['backward_nfe'].append(model.bm.sum)
    results['train_acc'].append(train_acc)
    results['val_acc'].append(val_acc)
    results['best_epoch'] = best_epoch
    results['best_val_acc'] = best_val_acc

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}'
    print(log.format(epoch, results['time'][-1], results['loss'][-1], results['forward_nfe'][-1], results['backward_nfe'][-1], results['train_acc'][-1], results['val_acc'][-1]))

  print('best val accuracy {:03f} at epoch {:d}'.format(best_val_acc, best_epoch))

  # TODO: Save results
  # cora_epoch_101_adjoint_false_... . pickle
  pickle.dump( results, open( f"../results/{opt['dataset']}_{opt['method']}_stepsize_{opt['dt']}_run_{run_count}.pickle", "wb" ) )

  return train_acc, best_val_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1, help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations allowed.")
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true', help="multiply attention scores by edge weights before softmax")
  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001, help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")

  # Stefan's experiment args
  parser.add_argument('--count_runs', type=int, default=10,
                      help="number of runs to average results over per parameter settings for each experiment")

  args = parser.parse_args()
  opt = vars(args)
  opt = get_cora_opt(opt)

  opt['epoch'] = 31
  opt['adjoint'] = True
  #opt['method'] = 'explicit_adams'
  opt['method'] = 'implicit_adams'
  #opt['method'] = 'dopri5'
  opt['adjoint_method'] = opt['method']
  opt['max_iters'] = 10000
  opt['step_size'] = opt['dt_min'] = 0.01
  opt['tol_scale'] = 100.0
  opt['tol_scale_adjoint'] = 100.0

  # DEBUG
  #for k in ['dataset', 'epoch', 'adjoint', 'rewiring', 'adaptive', 'dt', 'dt_min', 'method', 'adjoint_method']:
  #  print(k, opt[k])
  #main(opt, 0)

  # Run combination of experiments
  for stepsize in [0.5, 0.25, 0.1, 0.01]: # 2.0, 1.0
    print(f'*** Doing stepsize {stepsize} ***')
    for idx in range(opt['count_runs']):
      print(f'*** Doing run {idx} ***')
      # NOTE: I think setting dt_min may not be necessary, doing it just to be safe!
      opt['step_size'] = opt['dt_min'] = stepsize
      main(opt, idx)
