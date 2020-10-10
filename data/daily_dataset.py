import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils import utils


class StockSeqDataset(Dataset):
  """Pytorch dataset that breaks sequence data into windows of specified size."""
  def __init__(self, df, cols, window_size):
    self.df = df
    self.cols = cols
    self.window_size = window_size

  def __getitem__(self, index):
    assert 0 <= index < len(self)
    x_end = index + self.window_size
    tmap = {}
    for col in self.cols:
      x = self.df.iloc[index:x_end][col]
      tmap[col] = th.from_numpy(np.array(x, dtype=np.float32))
    return tmap

  def __len__(self):
    return len(self.df) - self.window_size + 1

