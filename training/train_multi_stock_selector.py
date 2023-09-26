import shutil
import os

import numpy as np
import torch as th
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from data import daily_dataset
from data import daily_df
from data import date_range
from models import multi_stock_selector
from utils import checkpoint

batch_size = 128
hist_win = 8

train_date_range = date_range.DateRange('2003-01-01', '2017-01-01')
eval_date_range = date_range.DateRange('2017-01-01', '2020-01-01')

tickers = ('GOOG', 'AAPL', 'MSFT', 'SPY', 'SHY')
num_tickers = len(tickers)

CKPT_BASE = '/media/14tb/ml/models/zetaqubit/trading/multi_stock_selector'
exp_name = 'v1'
exp_name += '.' + '_'.join(tickers) + '_' + train_date_range.start + ':' + train_date_range.end
exp_dir = os.path.join(CKPT_BASE, exp_name)

resume = False
if not resume:
    shutil.rmtree(exp_dir, ignore_errors=True)
os.makedirs(exp_dir, exist_ok=True)

def load_df(tickers, date_range):
  cfg = daily_df.Cfg(tickers=tickers)
  df = daily_df.load_df(cfg, date_range)
  print(f'Loaded {len(df)} days.')
  df = df.dropna(how='any').reset_index(drop=True)
  print(f'Kept {len(df)} days.')
  return df

df_train = load_df(tickers, train_date_range)
df_valid = load_df(tickers, eval_date_range)


PRICE_REPR = daily_dataset.PriceRepr.RETURN

def create_ds_dl(df, tickers, price_repr=PRICE_REPR):
  ds = daily_dataset.StockSeqDataset(
    df=df,
    cols=[f'{ticker}_adjclose' for ticker in tickers],
    window_size=hist_win + 1,
    price_repr=price_repr)

  ds_multistock = daily_dataset.MultiStockSelectionDataset(ds)
  dl_multistock = DataLoader(dataset=ds_multistock, batch_size=batch_size,
                            shuffle=True)
  return ds_multistock, dl_multistock

ds_train, dl_train = create_ds_dl(df_train, tickers)
ds_valid, dl_valid = create_ds_dl(df_valid, tickers)

model = multi_stock_selector.MultiStockSelector(
    num_tickers=num_tickers, hist_win=hist_win)
loss_fn = th.nn.CrossEntropyLoss()

learning_rate = 3e-4
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
lr_schedule = th.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)


def eval(model, dl):
  model.eval()
  with th.no_grad():
    loss_avg = 0
    epoch_accuracy = 0
    assert PRICE_REPR == daily_dataset.PriceRepr.RETURN
    log_return = 0
    base_log_return = [0] * num_tickers
    oracle_log_return = 0
    for batch in dl:
        x = batch['x']
        label = batch['label']
        logits = model(x)
        loss = loss_fn(logits, label)
        loss_avg += loss.item()
        preds = th.argmax(logits, axis=-1)
        accuracy = (label == preds).sum().item() / label.shape[0]
        epoch_accuracy += accuracy
        y = batch['y']
        day_return = y[np.arange(y.shape[0]), preds]
        log_return += day_return.log().sum()
        for i in range(num_tickers):
          ret = y[:, i]
          base_log_return[i] += ret.log().sum()
        oracle = y.max(dim=-1)[0]
        oracle_log_return += oracle.log().sum()
    print(f'valid loss: {loss_avg / len(dl):.4f}; '
          f'accuracy: {100 * epoch_accuracy / len(dl):.2f}%; '
          f'total_return: {log_return.exp():.2f}; ' +
          ''.join(
            f'{tickers[i]}_return: {base_log_return[i].exp():.2f}; '
            for i in range(num_tickers)
          ) +
          f'oracle_return: {oracle_log_return.exp():.2f}; '
    )
  model.train()


save_interval = 10
epochs = 100
for epoch in range(epochs):
    loss_avg = 0
    epoch_accuracy = 0
    log_return = 0
    for batch in dl_train:
        x = batch['x']
        label = batch['label']
        logits = model(x)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_avg += loss.item()
        preds = th.argmax(logits, axis=-1)
        accuracy = (label == preds).sum().item() / label.shape[0]
        epoch_accuracy += accuracy
    lr_schedule.step()
    print(f'epoch: {epoch}; '
          f'lr: {lr_schedule.get_last_lr()[0]:.5f}; '
          f'train loss: {loss_avg / len(dl_train):.4f}; '
          f'accuracy: {100 * epoch_accuracy / len(dl_train):.2f}%; ')
    eval(model, dl_train)
    eval(model, dl_valid)

    if epoch % save_interval == 0:
      checkpoint.save_ckpt(exp_dir, model, optimizer, step=epoch)
checkpoint.save_ckpt(exp_dir, model, optimizer, step=epoch)