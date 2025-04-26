from absl import flags
from absl import app
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env

FLAGS = flags.FLAGS
flags.DEFINE_string('instrument', 'yfinance-GOOG-1h', 'The instrument to trade (e.g. GOOG, BTC-USD)')

def main(argv):

    # Import your fresh data
    # df = pd.read_pickle("./data_cache/huobi-BTCUSDT-1m.pkl")
    df = pd.read_pickle(f"./data_cache/{FLAGS.instrument}.pkl")

    # df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change()

    # Create the feature : open[t] / close[t]
    df["feature_open"] = df["open"]/df["close"]

    # Create the feature : high[t] / close[t]
    df["feature_high"] = df["high"]/df["close"]

    # Create the feature : low[t] / close[t]
    df["feature_low"] = df["low"]/df["close"]

    # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()

    df.dropna(inplace= True) # Clean again !
    # Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

    env = gym.make("TradingEnv",
            name=FLAGS.instrument,
            df = df, # Your dataset with your custom features
            positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
            # trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
            borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        )
    env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )

    # Run an episode until it ends :
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        # position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        # position_index = np.random.choice([1, 2])
        if observation[1] < 1:
            position_index = 2
        else:
            position_index = 1
        observation, reward, done, truncated, info = env.step(position_index)

    env.save_for_render(dir="render_logs")

if __name__ == '__main__':
    app.run(main)