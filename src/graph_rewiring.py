"""
functions to generate a graph from the input graph and features
"""
import numpy as np
import numba
import torch
import torch.nn.functional as F
import torch_sparse
from torch_sparse import coalesce
from torch_scatter import scatter
from torch_geometric.transforms.two_hop import TwoHop
from torch_geometric.transforms import GDC
from utils import add_remaining_self_loops, get_rw_adj, get_full_adjacency, to_undirected, to_dense_adj, dense_to_sparse, ROOT_DIR
from pykeops.torch import LazyTensor
import os
import pickle
from distances_kNN import apply_dist_KNN, apply_dist_threshold, get_distances, apply_feat_KNN
from hyperbolic_distances import hyperbolize



def jit(**kwargs):
  def decorator(func):
    try:
      return numba.jit(cache=True, **kwargs)(func)
    except RuntimeError:
      return numba.jit(cache=False, **kwargs)(func)

  return decorator


###

def get_two_hop(data):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  th = TwoHop()
  data = th(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def apply_gdc(data, opt, type="combined"):
  # print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  # print('performing gdc transformation with method {}, sparsification {}'.format(opt['gdc_method'],
  #                                                                                opt['gdc_sparsification']))
  if opt['gdc_method'] == 'ppr':
    diff_args = dict(method='ppr', alpha=opt['ppr_alpha'])
  else:
    diff_args = dict(method='heat', t=opt['heat_time'])
  if opt['gdc_sparsification'] == 'topk':
    sparse_args = dict(method='topk', k=opt['gdc_k'], dim=0)
    diff_args['eps'] = opt['gdc_threshold']
  else:
    sparse_args = dict(method='threshold', eps=opt['gdc_threshold'])
    diff_args['eps'] = opt['gdc_threshold']
  print('gdc sparse args: {}'.format(sparse_args))
  if opt['self_loop_weight'] != 0:
    gdc = GDCWrapper(float(opt['self_loop_weight']),
                     normalization_in='sym',
                     normalization_out='col',
                     diffusion_kwargs=diff_args,
                     sparsification_kwargs=sparse_args, exact=opt['exact'])
  else:
    gdc = GDCWrapper(self_loop_weight=None,
                     normalization_in='sym',
                     normalization_out='col',
                     diffusion_kwargs=diff_args,
                     sparsification_kwargs=sparse_args, exact=opt['exact'])
  if isinstance(data.num_nodes, list):
    data.num_nodes = data.num_nodes[0]

  if type == 'combined':
    data = gdc(data)
  elif type == 'pos_encoding':
    if opt['pos_enc_orientation'] == "row":  # encode row of S_hat
      return gdc.position_encoding(data)
    elif opt['pos_enc_orientation'] == "col":  # encode col of S_hat
      return gdc.position_encoding(data).T

  # print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def make_symmetric(data):
  n = data.num_nodes
  if data.edge_attr is not None:
    ApAT_index = torch.cat([data.edge_index, data.edge_index[:, [1, 0], :]], dim=2).transpose(1, 2)
    ApAT_value = torch.cat([data.edge_attr, data.edge_attr], dim=1)
    
    edge_weights = to_dense_adj(ApAT_index, ApAT_value)
    sorted, _ = torch.sort(ApAT_index, dim=1, descending=True)
    ApAT_index = torch.unique_consecutive(sorted, dim=1).transpose(1, 2).long()
    index0 = torch.arange(ApAT_index.shape[0])[:, None].expand(ApAT_index.shape[0], ApAT_index.shape[2])
    index1 = ApAT_index[:, 0, :]
    index2 = ApAT_index[:, 1, :]
    ApAT_value = edge_weights[index0, index1, index2]
    # ei, ew = coalesce(ApAT_index, ApAT_value, n, n, op="add")
    ei, ew = ApAT_index, ApAT_value
  else:
    ei = to_undirected(data.edge_index)
    ew = None

  ei, ew = get_rw_adj(ei, edge_weight=ew, norm_dim=1, fill_value=0., num_nodes=n)

  return ei, ew


def dirichlet_energy(edge_index, edge_weight, n, X):
  if edge_weight is None:
    edge_weight = torch.ones(edge_index.size(0), edge_index.size(2),
                             device=edge_index.device)
  index0 = torch.arange(edge_index.shape[0])[:, None].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  index1 = edge_index[:, 0].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  index2 = edge_index[:, 1].expand(edge_index.shape[0], edge_index.shape[2]).flatten()
  indices = torch.stack([index0, index1, index2] , dim=0)
  adj_mat = torch.sparse_coo_tensor(indices, edge_weight.flatten(), [X.shape[0], X.shape[1], X.shape[1]],
                                       requires_grad=True).to(edge_weight.device)
  de = torch.matmul(adj_mat.to_dense(), X)
  return torch.matmul(X.transpose(1, 2), de)


def KNN(x, opt):
  # https://github.com/getkeops/keops/tree/3efd428b55c724b12f23982c06de00bc4d02d903
  k = opt['rewire_KNN_k']
  print(f"Rewiring with KNN: t={opt['rewire_KNN_T']}, k={opt['rewire_KNN_k']}")
  
  x_cloned = torch.clone(x)
  mask = torch.all(x_cloned == torch.zeros(x.shape[-1]), dim=-1)
  x_cloned[mask]= torch.nan
  x_cloned[torch.isnan(x_cloned)] = float('inf') 
  
  X_i = LazyTensor(x_cloned[:, :, None, :])  # (B, N, 1, hd)
  X_j = LazyTensor(x_cloned[:, None, :, :])  # (B, 1, N, hd)

  # distance between all the grid points and all the random data points
  D_ij = ((X_i - X_j) ** 2).sum(-1)
  # take the indices of the K closest neighbours measured in euclidean distance
  indKNN = D_ij.argKmin(k, dim=2).reshape(x.shape[0], -1)[:, None, :]
  first_index = torch.arange(x.shape[1]).repeat(k, 1).transpose(0, 1).flatten().repeat(x.shape[0], 1)[:, None, :]
  # LS = torch.linspace(0, len(indKNN.view(-1)), len(indKNN.view(-1)) + 1, dtype=torch.int64, device=indKNN.device)[
  #      :-1].unsqueeze(0) // k
  # ei = torch.cat([LS, indKNN.view(1, -1)], dim=0)
  ei = torch.cat([first_index, indKNN], dim=1)

  if opt['rewire_KNN_sym']:
    ei = to_undirected(ei)

  return ei


@torch.no_grad()
def apply_KNN(x, graph_data, x2, pos_encoding, model, opt):
  if opt['rewire_KNN_T'] == "raw":
    ei = KNN(x, opt)  # rewiring on raw features here
  elif opt['rewire_KNN_T'] == "T0":
    ei = KNN(model.forward_encoder(x, pos_encoding), opt)
  elif opt['rewire_KNN_T'] == 'TN':
    ei = KNN(model.forward_ODE(x, graph_data, pos_encoding, x2), opt)
  else:
    raise Exception("Need to set rewire_KNN_T")
  return ei


def edge_sampling(model, graph_data, z, opt):
  if opt['edge_sampling_space'] == 'attention':
    attention_weights = model.odeblock.get_attention_weights(z)
    mean_att = attention_weights.mean(dim=1, keepdim=False)
    threshold = torch.quantile(mean_att, opt['edge_sampling_rmv'])
    mask = mean_att > threshold

    threshold = torch.quantile(mean_att, opt['edge_sampling_rmv'])
    mask = mean_att >= threshold
  elif opt['edge_sampling_space'] in ['pos_distance', 'z_distance', 'pos_distance_QK', 'z_distance_QK']:
    temp_att_type = model.opt['attention_type']
    model.opt['attention_type'] = model.opt[
      'edge_sampling_space']  # this changes the opt at all levels as opt is assigment link
    pos_enc_distances = model.odeblock.get_attention_weights(z)  # forward pass of multihead_att_layer
    model.odeblock.multihead_att_layer.opt['attention_type'] = temp_att_type
    # threshold on distances so we throw away the biggest, opposite to attentions
    threshold = torch.quantile(pos_enc_distances, 1 - opt['edge_sampling_rmv'])
    mask = pos_enc_distances < threshold

  index0 = torch.arange(z.shape[0]).unsqueeze(1).unsqueeze(2)
  index1 = torch.arange(2).unsqueeze(0).unsqueeze(2)
  index2 = mask.unsqueeze(1)
  # model.odeblock.odefunc.edge_index = model.odeblock.odefunc.edge_index[:, mask.T]
  graph_data.edge_index = graph_data.edge_index[index0, index1, index2]

  if opt['edge_sampling_sym']:
    graph_data.edge_index = to_undirected(graph_data.edge_index)

  return graph_data.edge_index


@torch.no_grad()
def add_outgoing_attention_edges(model, graph_data, M):
  """
  add new edges for nodes that other nodes tend to pay attention to
  :params M: The number of edges to add. 2 * M get added to the edges index to make them undirected
  """
  atts = model.odeblock.odefunc.attention_weights.mean(dim=2)
  dst = graph_data.edge_index[:, 1, :]

  importance = scatter(atts, dst, dim=1, dim_size=model.num_nodes,
                       reduce='sum').to(model.device)  # column sum to represent outgoing importance
  degree = scatter(torch.ones(size=atts.shape, device=model.device), dst, dim=1, dim_size=model.num_nodes,
                   reduce='sum')
  normed_importance = torch.divide(importance, degree)
  # todo squareplus might be better here.
  importance_probs = F.softmax(normed_importance, dim=1).to(model.device)
  anchors = torch.multinomial(importance_probs, M, replacement=True).to(model.device)
  anchors2 = torch.multinomial(torch.ones(graph_data.num_nodes, device=model.device), M, replacement=True).to(model.device)

  new_edges = torch.cat([torch.stack([anchors, anchors2], dim=1), torch.stack([anchors2, anchors], dim=1)], dim=2)
  return new_edges


@torch.no_grad()
def add_edges(model, graph_data, opt):
  num_nodes = graph_data.num_nodes
  num_edges = graph_data.edge_index.shape[2]
  M = int(num_edges * opt['edge_sampling_add'])
  # generate new edges and add to edge_index
  if opt['edge_sampling_add_type'] == 'random':
    batch_size = graph_data.edge_index.shape[0]
    new_edges = np.random.choice(num_nodes, size=(batch_size, 2, M), replace=True, p=None)
    new_edges = torch.tensor(new_edges, device=model.device)
    new_edges2 = new_edges[:,[1, 0], :]
    cat = torch.cat([graph_data.edge_index, new_edges, new_edges2], dim=2)
  elif opt['edge_sampling_add_type'] == 'anchored':
    pass
  elif opt['edge_sampling_add_type'] == 'importance':
    if M > 0:
      new_edges = add_outgoing_attention_edges(model, graph_data, M)
      cat = torch.cat([graph_data.edge_index, new_edges], dim=model.odeblock.odefunc2)
    else:
      cat = graph_data.edge_index
  elif opt['edge_sampling_add_type'] == 'degree':  # proportional to degree
    pass
  elif opt['edge_sampling_add_type'] == 'n2_radius':
    return get_full_adjacency(graph_data.shape[0], num_nodes)
  
  n = num_nodes
  index0 = torch.arange(cat.shape[0])[:, None].expand(cat.shape[0], cat.shape[2]).flatten()
  index1 = cat[:,0].flatten()
  index2 = cat[:,1].flatten()
  indices = torch.stack([index0, index1, index2] , dim=0)
  edge_mat = torch.sparse_coo_tensor(indices, torch.ones(cat.shape[0], cat.shape(2)), [cat.shape[0], n, n],
                                     requires_grad=True).to(model.device).to_dense()
  nonzero_mask = torch.zeros_like(edge_mat)
  nonzero_mask[edge_mat != 0] = 1
  nonzero_mask = nonzero_mask.reshape(-1, n*n).unsqueeze(2)
  new_edges = torch.cartesian_prod(torch.arange(n), torch.arange(n))[None,:,:].expand(cat.shape[0], n*n, 2)
  new_edges = new_edges * nonzero_mask
  sorted, _ = torch.sort(new_edges, dim=1, descending=True)
  new_ei = torch.unique_consecutive(sorted, dim=1).transpose(1,2)
  return new_ei


@torch.no_grad()
def apply_edge_sampling(x, pos_encoding, model, graph_data, opt, x2=None):
  print(f"Rewiring with edge sampling")

  # add to model edge index
  graph_data.edge_index = add_edges(model, graph_data, opt)

  # get Z_T0 or Z_TN
  if opt['edge_sampling_T'] == "T0":
    z = model.forward_encoder(x, pos_encoding)
  elif opt['edge_sampling_T'] == 'TN':
    z = model.forward_ODE(x, graph_data, pos_encoding, x2)

  # sample the edges and update edge index in model
  edge_sampling(model, graph_data, z, opt)


def apply_beltrami(data, opt, data_dir=f'{ROOT_DIR}/data'):
  pos_enc_dir = os.path.join(f"{data_dir}", "pos_encodings")
  # generate new positional encodings
  # do encodings already exist on disk?
  fname = os.path.join(pos_enc_dir, f"{opt['dataset']}_{opt['pos_enc_type']}.pkl")
  print(f"[i] Looking for positional encodings in {fname}...")

  # - if so, just load them
  if os.path.exists(fname):
    print("    Found them! Loading cached version")
    with open(fname, "rb") as f:
      # pos_encoding = pickle.load(f)
      pos_encoding = pickle.load(f)
    if opt['pos_enc_type'].startswith("DW"):
      pos_encoding = pos_encoding['data']

  # - otherwise, calculate...
  else:
    print("    Encodings not found! Calculating and caching them")
    # choose different functions for different positional encodings
    if opt['pos_enc_type'] == "GDC":
      pos_encoding = apply_gdc(data, opt, type="pos_encoding")
    else:
      print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
      quit()
    # - ... and store them on disk
    POS_ENC_PATH = os.path.join(data_dir, "pos_encodings")
    if not os.path.exists(POS_ENC_PATH):
      os.makedirs(POS_ENC_PATH)

    if opt['pos_enc_csv']:
      sp = pos_encoding.to_sparse()
      table_mat = np.concatenate([sp.indices(), np.atleast_2d(sp.values())], axis=0).T
      np.savetxt(f"{fname[:-4]}.csv", table_mat, delimiter=",")

    with open(fname, "wb") as f:
      pickle.dump(pos_encoding, f)

  return pos_encoding


def apply_pos_dist_rewire(data, opt, data_dir='../data'):
  if opt['pos_enc_type'].startswith("HYP"):
    pos_enc_dir = os.path.join(f"{data_dir}", "pos_encodings")
    # generate new positional encodings distances
    # do encodings already exist on disk?
    fname = os.path.join(pos_enc_dir, f"{opt['dataset']}_{opt['pos_enc_type']}_dists.pkl")
    print(f"[i] Looking for positional encoding DISTANCES in {fname}...")

    # - if so, just load them
    if os.path.exists(fname):
      print("    Found them! Loading cached version")
      with open(fname, "rb") as f:
        pos_dist = pickle.load(f)
      # if opt['pos_enc_type'].startswith("DW"):
      #   pos_dist = pos_dist['data']

    # - otherwise, calculate...
    else:
      print("    Encodings not found! Calculating and caching them")
      # choose different functions for different positional encodings
      if opt['pos_enc_type'].startswith("HYP"):
        pos_encoding = apply_beltrami(data, opt)
        pos_dist = hyperbolize(pos_encoding)


      else:
        print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
        quit()
      # - ... and store them on disk
      POS_ENC_PATH = os.path.join(data_dir, "pos_encodings")
      if not os.path.exists(POS_ENC_PATH):
        os.makedirs(POS_ENC_PATH)

      # if opt['pos_enc_csv']:
      #   sp = pos_encoding.to_sparse()
      #   table_mat = np.concatenate([sp.indices(), np.atleast_2d(sp.values())], axis=0).T
      #   np.savetxt(f"{fname[:-4]}.csv", table_mat, delimiter=",")

      with open(fname, "wb") as f:
        pickle.dump(pos_dist, f)

      if opt['gdc_sparsification'] == 'topk':
        ei = apply_dist_KNN(pos_dist, opt['gdc_k'])
      elif opt['gdc_sparsification'] == 'threshold':
        ei = apply_dist_threshold(pos_dist, opt['pos_dist_quantile'])

  elif opt['pos_enc_type'].startswith("DW"):
    pos_encoding = apply_beltrami(data, opt, data_dir)
    if opt['gdc_sparsification'] == 'topk':
      ei = apply_feat_KNN(pos_encoding, opt['gdc_k'])
      # ei = KNN(pos_encoding, opt)
    elif opt['gdc_sparsification'] == 'threshold':
      dist = get_distances(pos_encoding)
      ei = apply_dist_threshold(dist)

  data.edge_index = torch.from_numpy(ei).type(torch.LongTensor)

  return data


class GDCWrapper(GDC):
  def __init__(self, self_loop_weight=1, normalization_in='sym',
               normalization_out='col',
               diffusion_kwargs=dict(method='ppr', alpha=0.15),
               sparsification_kwargs=dict(method='threshold',
                                          avg_degree=64), exact=True):
    super(GDCWrapper, self).__init__(self_loop_weight, normalization_in, normalization_out, diffusion_kwargs,
                              sparsification_kwargs, exact)
    self.self_loop_weight = self_loop_weight
    self.normalization_in = normalization_in
    self.normalization_out = normalization_out
    self.diffusion_kwargs = diffusion_kwargs
    self.sparsification_kwargs = sparsification_kwargs
    self.exact = exact

    if self_loop_weight:
      assert exact or self_loop_weight == 1

  def position_encoding(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    # edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                             **self.diffusion_kwargs)
      edge_index, edge_weight = dense_to_sparse(diff_mat)
      # edge_index, edge_weight = self.sparsify_dense(
      #   diff_mat, **self.sparsification_kwargs)
    else:
      edge_index, edge_weight = self.diffusion_matrix_approx(
        edge_index, edge_weight, N, self.normalization_in,
        **self.diffusion_kwargs)
      # edge_index, edge_weight = self.sparsify_sparse(
      #   edge_index, edge_weight, N, **self.sparsification_kwargs)

    # edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    return to_dense_adj(edge_index,
                        edge_attr=edge_weight).squeeze()
