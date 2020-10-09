"""Tests for data.daily_dataset."""

import unittest

from data import daily_dataset as ds
from data import date_range as dr


class DailyDatasetTest(unittest.TestCase):

  def test_load_df_one_ticker(self):
    df = ds.load_df_one_ticker('GOOG')
    self.assertEqual(
        list(df.columns),
        ['date', 'volume', 'open', 'high', 'low', 'close', 'adjclose'])
    self.assertGreater(len(df), 3990)
    self.assertFalse(df.isnull().any().any())
    self.assertEqual(df.date.dt.strftime('%Y-%m-%d').iloc[0], '2004-08-19')

  def test_load_df_one_ticker_date_range(self):
    df = ds.load_df_one_ticker('GOOG', dr.DateRange('2020-01-06', '2020-01-10'))
    self.assertEqual(
        list(df.date.dt.strftime('%Y-%m-%d')),
        ['2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09'])
    
  def test_load_df_one(self):
    df = ds.load_df(ds.Cfg(['GOOG']))
    cols = ['volume', 'open', 'high', 'low', 'close', 'adjclose']
    self.assertEqual(list(df.columns), ['date'] + ['GOOG_' + col for col in cols])
    self.assertGreater(len(df), 3990)
    self.assertFalse(df.isnull().any().any())

  def test_load_df_multiple(self):
    df = ds.load_df(ds.Cfg(['GOOG', 'FB']))
    cols = ['volume', 'open', 'high', 'low', 'close', 'adjclose']
    self.assertEqual(
        list(df.columns),
        ['date'] + ['GOOG_' + col for col in cols] + ['FB_' + col for col in cols])
    self.assertGreater(len(df), 3990)

    df_fb = df.dropna()
    self.assertEqual(df_fb.date.dt.strftime('%Y-%m-%d').iloc[0], '2012-05-18')

  def test_load_df_date_range(self):
    df = ds.load_df(ds.Cfg(['GOOG', 'FB']), dr.DateRange('2020-01-06', '2020-01-10'))
    self.assertEqual(
        list(df.date.dt.strftime('%Y-%m-%d')),
        ['2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09'])
    self.assertFalse(df.isnull().any().any())
