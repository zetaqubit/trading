from absl import flags
from absl import app
import yfinance as yf
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('instrument', 'GOOG', 'The instrument to download (e.g. GOOG, AAPL)')
flags.DEFINE_string('interval', '1h', 'The interval to download (e.g. 1h, 1d)')
flags.DEFINE_string('period', '60d', 'The period to download (e.g. 60d, 1y)')
flags.DEFINE_string('output_fmt', 'csv', 'Output format to use. [csv, pkl]')
flags.DEFINE_string('output_dir', '~/data/zetaqubit/stock/yfinance', 'Output root dir.')

def main(argv):
    # Download data at specified interval
    data = yf.download(FLAGS.instrument, interval=FLAGS.interval, period=FLAGS.period)

    # Keep only required columns and rename them
    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    # Check if index is timezone-aware
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(df.head())

    if FLAGS.output_fmt == 'csv':
        output_file = f'{FLAGS.output_dir}/{FLAGS.instrument}-{FLAGS.interval}.csv'
        df.to_csv(output_file)
    elif FLAGS.output_fmt == 'pkl':
        output_file = f'{FLAGS.output_dir}/{FLAGS.instrument}-{FLAGS.interval}.pkl'
        df.to_pickle(output_file)
    else:
        raise NotImplementedError(f'Unknown {FLAGS.output_fmt=}.')
    print(f"Saved {len(df)} rows to {output_file}.")

if __name__ == '__main__':
    app.run(main)