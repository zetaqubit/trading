"""Tests for utils.utils."""

import unittest

import numpy as np

from utils import utils


class UtilsTest(unittest.TestCase):

  def test_split_xy_basic(self):
    items = np.array([1, 2, 3, 4, 5])
    actual = utils.split_xy(items, 2)
    actual = tuple(map(list, actual))
    self.assertEqual(
        actual,
        ([1, 2, 3], [4, 5]))

  def test_split_xy_dict(self):
    items = {
        'a': np.array([1, 2, 3]),
        'b': np.array([4, 5, 6, 7]),
    }
    actual = utils.split_xy(items, 1)
    actual = (
        {k: list(v) for k, v in actual[0].items()},
        {k: list(v) for k, v in actual[1].items()},
    )
    self.assertEqual(
        actual,
        (
          { 'a': [1, 2], 'b': [4, 5, 6] },
          { 'a': [3], 'b': [7] },
        )
    )

