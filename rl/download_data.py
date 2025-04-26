from gym_trading_env.downloader import download
import datetime
import pandas as pd

# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
download(exchange_names = ["huobi"],
    symbols= ["BTC/USDT"],
    timeframe= "1h",
    dir = "data_cache",
    since= datetime.datetime(year= 2020, month= 1, day=1),
)

# download(exchange_names = ["huobi"],
#     symbols= ["BTC/USDT"],
#     timeframe= "1m",
#     dir = "data",
#     since= datetime.datetime(year= 2025, month= 1, day=1),
# )