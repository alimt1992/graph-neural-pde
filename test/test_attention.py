#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
# needed for CI/CD
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn

from function_GAT_attention import SpGraphAttentionLayer, ODEFuncAtt
from utils import to_dense_adj, softmax
from data import get_dataset
from test_params import OPT


class AttentionTests(unittest.TestCase):
  def setUp(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with (torch.device(self.device)):
      self.edge = tensor([[[0, 2, 2, 1], [1, 0, 1, 2]]])
      self.x = tensor([[[1., 2.], [3., 2.], [4., 5.]]], dtype=torch.float)
      self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
      self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
      self.edge1 = tensor([[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]])
      self.x1 = torch.ones((1, 3, 2), dtype=torch.float)

      self.leakyrelu = nn.LeakyReLU(0.2)
    opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10,
           'attention_norm_idx': 0, 'add_source': False, 'max_nfe': 1000, 'mix_features': False, 'attention_dim': 32,
           'batch_siza': 1, 'multi_modal': False, 'mixed_block': False, 'rewiring': None, 'no_alpha_sigmoid': False,
           'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None, 'total_deriv': None,
           'directional_penalty': None}
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
    attention = softmax(edge_e, self.edge[:, 1])
    print(attention)

  def test_function(self):
    in_features = self.x.shape[2]
    out_features = self.x.shape[2]

    def get_round_sum(tens, n_digits=3):
      val = torch.sum(tens, dim=int(not self.opt['attention_norm_idx']) + 1)
      return (val * 10 ** n_digits).round() / (10 ** n_digits)

    with (torch.device(self.device)):
      att_layer = SpGraphAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(self.x, self.edge)  # should be n_edges x n_heads
      self.assertTrue(attention.shape == (self.edge.shape[0], self.edge.shape[2], self.opt['heads']))
      dense_attention1 = to_dense_adj(self.edge, edge_attr=attention[:, :, 0])
      dense_attention2 = to_dense_adj(self.edge, edge_attr=attention[:, :, 1])

      self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention1), 1.)))
      self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention2), 1.)))

      self.assertTrue(torch.all(attention > 0.))
      self.assertTrue(torch.all(attention <= 1.))

    dataset = get_dataset(self.opt, '../data', False)
    data = dataset.data
    in_features = data.x.shape[2]
    out_features = data.x.shape[2]
    data.x, data.edge_index = data.x.to(self.device), data.edge_index.to(self.device)

    with (torch.device(self.device)):
      att_layer = SpGraphAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(data.x, data.edge_index)  # should be n_edges x n_heads

      self.assertTrue(attention.shape == (data.edge_index.shape[0], data.edge_index.shape[2], self.opt['heads']))
      dense_attention1 = to_dense_adj(data.edge_index, edge_attr=attention[:, :, 0])
      dense_attention2 = to_dense_adj(data.edge_index, edge_attr=attention[:, :, 1])
      self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention1), 1.)))
      self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention2), 1.)))
      self.assertTrue(torch.all(attention > 0.))
      self.assertTrue(torch.all(attention <= 1.))

  def test_symmetric_attention(self):
    with (torch.device(self.device)):
      in_features = self.x1.shape[2]
      out_features = self.x1.shape[2]
      att_layer = SpGraphAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
      attention, _ = att_layer(self.x1, self.edge1)  # should be n_edges x n_heads

      self.assertTrue(torch.all(torch.eq(attention, 0.5 * torch.ones((self.edge1.shape[0], self.edge1.shape[2], self.x1.shape[2])))))

  def test_module(self):
    dataset = get_dataset(self.opt, '../data', False)
    t = 1
    out_dim = 6
    func = ODEFuncAtt(dataset.data.num_features, out_dim, self.opt, self.device)
    func.edge_index = dataset.data.edge_index.to(self.device)
    out = func(t, dataset.data.x.to(self.device))
    print(out.shape)
    self.assertTrue(out.shape == dataset.data.x.shape)


if __name__ == '__main__':
  AT = AttentionTests()
  AT.setUp()
  AT.test()
  AT.test_function()
  AT.test_symmetric_attention()
  AT.test_module()