"""Tests for data.daily_dataset."""

import unittest

import pandas as pd
import torch as th

from data import daily_dataset
from data import daily_df
from data import date_range as dr


class StockSeqDatasetTest(unittest.TestCase):

  def test_multiple_columns(self):
    df = pd.DataFrame({
      'a': [0, 1, 2, 3, 4],
      'b': [10.0, 11.0, 12.0, 13.0, 14.0],
    })

    ds = daily_dataset.StockSeqDataset(df, cols=['a', 'b'], window_size=3)

    self.assertEqual(len(ds), 3)
    self.assertTrue(th.equal(ds[0]['a'], th.FloatTensor([0, 1, 2])))
    self.assertTrue(th.equal(ds[0]['b'], th.FloatTensor([10, 11, 12])))
    self.assertTrue(th.equal(ds[1]['a'], th.FloatTensor([1, 2, 3])))
    self.assertTrue(th.equal(ds[1]['b'], th.FloatTensor([11, 12, 13])))
    self.assertTrue(th.equal(ds[2]['a'], th.FloatTensor([2, 3, 4])))
    self.assertTrue(th.equal(ds[2]['b'], th.FloatTensor([12, 13, 14])))

