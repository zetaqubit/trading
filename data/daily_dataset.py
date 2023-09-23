import enum

import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils import utils


class PriceRepr(enum.Enum):
  DOLLAR = 'dollar'
  RETURN = 'return'
  LOG_RETURN = 'log_return'


class StockSeqDataset(Dataset):
  """Pytorch dataset that breaks sequence data into windows of specified size."""
  def __init__(self, df, cols, window_size, price_repr=PriceRepr.DOLLAR):
    self.df = df
    self.cols = cols
    self.window_size = window_size
    self.price_repr = price_repr
    if price_repr in (PriceRepr.RETURN, PriceRepr.LOG_RETURN):
      self.window_size += 1  # need 1 more day for ratios

  def __getitem__(self, index):
    assert 0 <= index < len(self)
    x_end = index + self.window_size
    tmap = {}
    for col in self.cols:
      x = self.df.iloc[index:x_end][col].to_numpy(dtype=np.float32)
      if self.price_repr == PriceRepr.RETURN:
        x = x[1:] / x[:-1]
      elif self.price_repr == PriceRepr.LOG_RETURN:
        x = x[1:] / x[:-1]
        x = np.log(x)
      tmap[col] = th.from_numpy(x)
    return tmap

  def __len__(self):
    return len(self.df) - self.window_size + 1
    # return 4


class MultiStockSelectionDataset(Dataset):
  def __init__(self, stock_seq_ds):
    self.ds = stock_seq_ds

  def __getitem__(self, index):
    tmap = self.ds[index]
    timeseries = th.stack(
      [tmap[col] for col in self.ds.cols],
      dim=-1)  # [b, t, n_tickers]
    x = timeseries[:-1, :]  # [t, n_tickers]
    y = timeseries[-1, :]  # [n_tickers]
    if self.ds.price_repr == PriceRepr.DOLLAR:
      day_return = y / x[-1, :]
    else:
      day_return = y
    label = th.argmax(day_return, dim=-1)
    return {
      'x': x,
      'y': y,
      'label': label,
    }

  def __len__(self):
    return len(self.ds)
