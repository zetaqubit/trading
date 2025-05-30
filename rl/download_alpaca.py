from absl import flags
from absl import app
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

FLAGS = flags.FLAGS
flags.DEFINE_string('instrument', 'GOOG', 'The instrument to download (e.g. GOOG, AAPL)')
flags.DEFINE_string('interval', '1Hour', 'The interval to download (e.g. 1Hour, 1Day)')
flags.DEFINE_string('period', '1y', 'The period to download (e.g. 60d, 1y)')
flags.DEFINE_string('output_fmt', 'csv', 'Output format to use. [csv, pkl]')
flags.DEFINE_string('output_dir', '~/data/zetaqubit/stock/alpaca', 'Output root dir.')
flags.DEFINE_string('api_key', None, 'Alpaca API key')
flags.DEFINE_string('api_secret', None, 'Alpaca API secret')
flags.DEFINE_boolean('paper', True, 'Use paper trading API')

def get_timeframe(interval: str) -> TimeFrame:
    """Convert interval string to Alpaca TimeFrame object."""
    interval_map = {
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame.Minute,
        '15Min': TimeFrame.Minute,
        '1Hour': TimeFrame.Hour,
        '1Day': TimeFrame.Day,
    }
    return interval_map.get(interval, TimeFrame.Hour)

def get_start_date(period: str) -> datetime:
    """Convert period string to start date."""
    now = datetime.now()
    if period.endswith('d'):
        days = int(period[:-1])
        return now - timedelta(days=days)
    elif period.endswith('y'):
        years = int(period[:-1])
        return now - timedelta(days=years*365)
    else:
        raise ValueError(f"Invalid period format: {period}")

def main(argv):
    # Initialize Alpaca client
    client = StockHistoricalDataClient(
        api_key=FLAGS.api_key,
        secret_key=FLAGS.api_secret,
        # paper=FLAGS.paper
    )

    # Prepare request parameters
    timeframe = get_timeframe(FLAGS.interval)
    start_date = get_start_date(FLAGS.period)
    end_date = datetime.now()

    # Create request
    request_params = StockBarsRequest(
        symbol_or_symbols=FLAGS.instrument,
        timeframe=timeframe,
        start=start_date,
        end=end_date
    )

    # Get bars
    bars = client.get_stock_bars(request_params)

    # Convert to DataFrame
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(FLAGS.instrument, level='symbol')

    # Rename columns to match yfinance format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # Ensure timezone is removed
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(df.head())

    end_date_print = end_date.strftime("%Y-%m-%d")
    prefix = f'{FLAGS.output_dir}/{FLAGS.instrument}-{FLAGS.period}-{FLAGS.interval}-{end_date_print}'

    # Save to file
    if FLAGS.output_fmt == 'csv':
        output_file = prefix + '.csv'
        df.to_csv(output_file)
    elif FLAGS.output_fmt == 'pkl':
        output_file = prefix + '.pkl'
        df.to_pickle(output_file)
    else:
        raise NotImplementedError(f'Unknown {FLAGS.output_fmt=}.')
    print(f"Saved {len(df)} rows to {output_file}.")

if __name__ == '__main__':
    app.run(main)
