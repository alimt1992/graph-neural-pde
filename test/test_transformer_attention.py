#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
# from torch_geometric.utils import softmax

from function_transformer_attention import SpGraphTransAttentionLayer, ODEFuncTransformerAtt
from data import get_dataset
from test_params import OPT
from utils import to_dense_adj, softmax, ROOT_DIR

class AttentionTests(unittest.TestCase):
  def setUp(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with (torch.device(self.device)):
      self.edge = tensor([[[0, 2, 2, 1], [1, 0, 1, 2]]])
      self.x = tensor([[[1., 2.], [3., 2.], [4., 5.]]], dtype=torch.float)
      self.edge1 = tensor([[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]])
      self.x1 = torch.ones((1, 3, 2), dtype=torch.float)

      self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
      self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
      self.leakyrelu = nn.LeakyReLU(0.2)
    opt = {'dataset': 'Citeseer', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10,
           'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
           'hidden_dim': 6, 'linear_attention': True, 'augment': False, 'adjoint': False, 'tol_scale': 1, 'time': 1,
           'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler', 'mixed_block': True, 'max_nfe': 1000,
           'mix_features': False, 'attention_dim': 32, 'rewiring': None, 'batch_size': 1, 'multi_modal': False,
           'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None,
           'total_deriv': None, 'directional_penalty': None, 'beltrami': False}
    self.opt = {**OPT, **opt}
  def tearDown(self) -> None:
    pass

  def test(self):
    h = torch.matmul(self.x, self.W)

    index0 = torch.arange(h.shape[0]).unsqueeze(1).unsqueeze(2)
    index2 = torch.arange(h.shape[2]).unsqueeze(0).unsqueeze(0)
    edge_h = torch.cat((h[index0, self.edge[:, 0, :].unsqueeze(2), index2], h[index0, self.edge[:, 1, :].unsqueeze(2), index2]), dim=2)
    self.assertTrue(edge_h.shape == torch.Size([self.edge.shape[0], self.edge.shape[2], 2 * 2]))
    ah = self.alpha.matmul(edge_h.transpose(1,2)).transpose(1,2)
    self.assertTrue(ah.shape == torch.Size([self.edge.shape[0], self.edge.shape[2], 1]))
    edge_e = self.leakyrelu(ah)
    # attention = softmax(edge_e[0], self.edge[0, 1])
    attention = softmax(edge_e, self.edge[:, 1])
    print(attention)

  def test_function(self):
    with (torch.device(self.device)):
      in_features = self.x.shape[2]
      out_features = self.x.shape[2]
      att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(self.x, self.edge)  # should be n_edges x n_heads
      self.assertTrue(attention.shape == (self.edge.shape[0], self.edge.shape[2], self.opt['heads']))
      dense_attention1 = to_dense_adj(self.edge, edge_attr=attention[:, :, 0])
      dense_attention2 = to_dense_adj(self.edge, edge_attr=attention[:, :, 1])

      def get_round_sum(tens, n_digits=3):
        val = torch.sum(tens, dim=int(not self.opt['attention_norm_idx']) + 1)
        round_sum = (val * 10 ** n_digits).round() / (10 ** n_digits)
        print('round sum', round_sum)
        return round_sum

      self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention1), torch.ones(size=dense_attention1.shape))))
      self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention2), torch.ones(size=dense_attention1.shape))))
      self.assertTrue(torch.all(attention > 0.))
      self.assertTrue(torch.all(attention <= 1.))

    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', True)
    data = dataset.data
    in_features = data.x.shape[2]
    out_features = data.x.shape[2]
    data.x, data.edge_index = data.x.to(self.device), data.edge_index.to(self.device)

    with (torch.device(self.device)):
      att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(data.x, data.edge_index)  # should be n_edges x n_heads
      self.assertTrue(attention.shape == (1, data.edge_index.shape[2], self.opt['heads']))
      dense_attention1 = to_dense_adj(data.edge_index, edge_attr=attention[:, :, 0])
      dense_attention1 = to_dense_adj(data.edge_index, edge_attr=attention[:, :, 1])
      print('sums:', torch.sum(torch.isclose(dense_attention1, torch.ones(size=dense_attention1.shape))), dense_attention1.shape)
      print('da1', dense_attention1)
      print('da2', dense_attention2)
      self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention1), torch.ones(size=dense_attention1.shape))))
      self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention2), torch.ones(size=dense_attention2.shape))))
      self.assertTrue(torch.all(attention > 0.))
      self.assertTrue(torch.all(attention <= 1.))

  def test_symmetric_attention(self):
    with (torch.device(self.device)):
      in_features = self.x1.shape[2]
      out_features = self.x1.shape[2]
      att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(self.x1, self.edge1)  # should be n_edges x n_heads

      self.assertTrue(torch.all(torch.isclose(att_layer.Q.weight, att_layer.K.weight)))
      self.assertTrue(torch.all(torch.eq(attention, 0.5 * torch.ones((self.edge1.shape[2], self.x1.shape[2])))))

  def test_module(self):
    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', False)
    t = 1
    out_dim = 6
    func = ODEFuncTransformerAtt(dataset.data.num_features, out_dim, self.opt, self.device)
    func.edge_index = dataset.data.edge_index.to(self.device)
    out = func(t, dataset.data.x.to(self.device))
    print(out.shape)
    self.assertTrue(out.shape == dataset.data.x.shape)

  def test_head_aggregation(self):
    with (torch.device(self.device)):
      in_features = self.x.shape[2]
      out_features = self.x.shape[2]
      self.opt['head'] = 4
      att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(self.x, self.edge)
  
      index0 = torch.arange(self.edge.shape[0])[:, None, None].expand(self.edge.shape[0], self.edge.shape[2], self.opt['heads']).flatten()
      index1 = self.edge[:,0][:, :, None].expand(self.edge.shape[0], self.edge.shape[2], self.opt['heads']).flatten()
      index2 = self.edge[:,1][:, :, None].expand(self.edge.shape[0], self.edge.shape[2], self.opt['heads']).flatten()
      index3 = torch.arange(self.opt['heads'])[None, None, :].expand(self.edge.shape[0], self.edge.shape[2], self.opt['heads']).flatten()
      indices = torch.stack([index0, index1, index2, index3] , dim=0)
      sparse_att = torch.sparse_coo_tensor(indices, attention.flatten(), [self.x.shape[0], self.x.shape[1], self.x.shape[1], self.opt['heads']],
                                           requires_grad=True).to(self.device)
      ax1 = torch.mean(torch.matmul(sparse_att.permute(0,3,1,2).to_dense(), self.x), dim=1)
  
      mean_attention = attention.mean(dim=2)
      index0 = torch.arange(self.edge.shape[0])[:, None].expand(self.edge.shape[0], self.edge.shape[2]).flatten()
      index1 = self.edge[:,0].expand(self.edge.shape[0], self.edge.shape[2]).flatten()
      index2 = self.edge[:,1].expand(self.edge.shape[0], self.edge.shape[2]).flatten()
      indices = torch.stack([index0, index1, index2] , dim=0)
      sparse_att = torch.sparse_coo_tensor(indices, mean_attention.flatten(), [self.x.shape[0], self.x.shape[1], self.x.shape[1]],
                                           requires_grad=True).to(self.device)
      ax2 = torch.matmul(sparse_att.to_dense(), self.x)
      self.assertTrue(torch.all(torch.isclose(ax1,ax2)))

  def test_two_way_edge(self):
    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', False)
    edge = dataset.data.edge_index
    dataset.data.x = dataset.data.x.squeeze()
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.edge_index = dataset.data.edge_index.squeeze()
    print(f"is_undirected {dataset.data.is_undirected()}")
    dataset.data.x = dataset.data.x.unsqueeze(0)
    dataset.data.y = dataset.data.y.unsqueeze(0)
    dataset.data.edge_index = dataset.data.edge_index.unsqueeze(0)

    edge_dict = {}

    for idx, src in enumerate(edge[0, 0, :]):
      src = int(src)
      if src in edge_dict:
        edge_dict[src].add(int(edge[0, 1, idx]))
      else:
        edge_dict[src] = set([int(edge[0, 1, idx])])

    print(f"edge shape {edge.shape}")
    src_test = edge[0, :, edge[0, 0, :] == 1][1, :]
    dst_test = edge[0, :, edge[0, 1, :] == 1][0, :]
    print('dst where src = 1', src_test)
    print('src where dst = 1', dst_test)

    for idx, dst in enumerate(edge[0, 1, :]):
      dst = int(dst)
      self.assertTrue(int(edge[0, 0, idx]) in edge_dict[dst])


if __name__ == '__main__':
  AT = AttentionTests()
  AT.setUp()
  AT.test_symmetric_attention()
  AT.test()
  AT.test_function()
  AT.test_module()
  AT.test_head_aggregation()
  AT.test_two_way_edge()