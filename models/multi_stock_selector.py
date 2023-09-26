import torch as th

class MultiStockSelector(th.nn.Module):
  def __init__(self, num_tickers, hist_win):
    super().__init__()
    self.ff = th.nn.Sequential(
      th.nn.LayerNorm(num_tickers * hist_win),
      th.nn.Linear(num_tickers * hist_win, 1000),
      th.nn.Sigmoid(),
      th.nn.Linear(1000, 1000),
      th.nn.Sigmoid(),
      th.nn.Linear(1000, num_tickers),
    )
    self.apply(self.init_)

  def init_(self, m):
    if isinstance(m, th.nn.Linear):
      th.nn.init.xavier_normal_(m.weight)
      th.nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    x = x.view(x.shape[0], -1)  # [b, t * n_tickers]
    logits = self.ff(x)
    return logits