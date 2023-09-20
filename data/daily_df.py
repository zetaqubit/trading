from dataclasses import dataclass
import re
from typing import List

import pandas as pd

from data import date_range as dr

COL_DATE = 'date'

@dataclass
class Cfg:
  tickers: List
  root_dir: str = '~/data/zetaqubit/trading/kaggle/full_history/'


def load_df_one_ticker(ticker, date_range: dr.DateRange = None):
  """Loads the DataFrame for ticker between [start_date, end_date)."""
  df = pd.read_csv(f'~/data/zetaqubit/trading/kaggle/full_history/{ticker}.csv')
  df[COL_DATE] = pd.to_datetime(df[COL_DATE])
  if date_range:
    if date_range.start:
      df = df.query(f'{COL_DATE} >= "{date_range.start}"')
    if date_range.end:
      df = df.query(f'{COL_DATE} < "{date_range.end}"')

  # reverse so oldest is first
  df = df.iloc[::-1].reset_index(drop=True)
  return df


def load_df(cfg: Cfg, date_range: dr.DateRange = None):
  """Loads the DataFrame with the specified tickers, between [start, end).

  The tickers are prepended to each column.
  """
  df_all = None
  for ticker in cfg.tickers:
    df = load_df_one_ticker(ticker, date_range)
    df = _add_ticker(df, ticker)
    if df_all is None:
      df_all = df
    else:
      df_all = pd.merge(df_all, df, how='outer', on=COL_DATE, suffixes=['', ''])
  return df_all


def _add_ticker(df, ticker):
  df = df.copy()
  df.columns = [
      ticker + '_' + col if col != COL_DATE else col
      for col in df.columns
  ]
  return df


def _remove_ticker(df):
  df = df.copy()
  # Assumes tickers are all uppercase, followed by '_'.
  df.columns = [re.sub('^[A-Z]*_', '', col) for col in df.columns]
  return df
