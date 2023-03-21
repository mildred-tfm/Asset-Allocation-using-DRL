# The following codes are largely inspired by this Github repository:
# https://github.com/Musonda2day/Asset-Portfolio-Management-usingDeep-Reinforcement-Learning-

from config import config
import yfinance as yf
import numpy as np
from models import DRLAgent
import pandas as pd
from pyfolio import timeseries
import matplotlib.pylab as plt
from backtest import  backtest_strat
import matplotlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython.display import display, HTML
from env_portfolio import StockPortfolioEnv
matplotlib.use('TkAgg')
matplotlib.use('Agg')
yf.pdr_override()
import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
os.chdir('/Users/yifei/Documents/Yifei_Stanford/Winter_2/CME_241/Project/Asset-Portfolio-Management-usingDeep-Reinforcement-Learning--main')

stock_chosen = set(['AAPL', 'NKE', 'BAC', 'AMZN', 'HD'])
#Select Stocks
stocks = config.STOCK_TICKER
ticker_list = config.STOCK_TICKER

#load stock data
import pickle
from finrl.preprocessing.data import data_split
from datetime import datetime

train_start_date = datetime(2008, 1, 1)
train_end_date = datetime(2018, 12, 31)
test_start_date = datetime(2019, 1, 1)
test_end_date = datetime(2020, 12, 31)

tech_indicator_list = ["close", "high", "open", "low", "volume"]
with open('data_ftr.pkl', 'rb') as handle:
    data_df = pickle.load(handle)

stock_dimension = len(data_df.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# split data set
train_data = data_split(data_df, train_start_date, train_end_date)
test_data = data_split(data_df, test_start_date, test_end_date)

train_df = train_data
test_df = test_data

stock_dimension = len(train_df.tic.unique())
state_space = stock_dimension

# Define a Function for Displaying the Cleaned Weights
def show_clean_p(port_df):
    p1_show_1 = (port_df.transpose()[0]).map(lambda x: "{:.3%}".format(x)).to_frame().transpose()
    return display(HTML(p1_show_1.to_html()))

# define equally weighted portfolios
ticker_list = list(train_df.columns) # Get List of all ticker symbols
n_assets = len(ticker_list) # Number of assets

weights_initial = [1/stock_dimension]*stock_dimension

# create trading environment
env_kwargs = {
    "hmax": 500,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicator_list,
    "action_space": stock_dimension,
    "reward_scaling": 0,
    'initial_weights': [1/stock_dimension]*stock_dimension
}
e_train_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# initialize agents
agent = DRLAgent(env = env_train)

# A2C agent
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name='a2c',
                                total_timesteps=50000)

# PPO Agent
agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=50000)

# DDPG Agent
agent = DRLAgent(env = env_train)
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}


model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
trained_ddpg = agent.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)

# A2C Train Model
e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

a2c_train_daily_return, a2c_train_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                        test_data = train_df,
                        test_env = env_trade,
                        test_obs = obs_trade)
# PPO Train Model
e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

ppo_train_daily_return, ppo_train_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                        test_data = train_df,
                        test_env = env_trade,
                        test_obs = obs_trade)
# DDPG Train Model
e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

ddpg_train_daily_return, ddpg_train_weights = DRLAgent.DRL_prediction(model=trained_ddpg,
                        test_data = train_df,
                        test_env = env_trade,
                        test_obs = obs_trade)

# A2C Test Model
e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

a2c_test_daily_return, a2c_test_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                        test_data = test_df,
                        test_env = env_trade,
                        test_obs = obs_trade)

a2c_test_weights.to_csv('a2c_test_weights.csv')

# PPO Test Model
e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

ppo_test_daily_return, ppo_test_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                        test_data = test_df,
                        test_env = env_trade,
                        test_obs = obs_trade)
ppo_test_weights.to_csv('ppo_test_weights.csv')

# DDPG Test Model
e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()
#
ddpg_test_daily_return, ddpg_test_weights = DRLAgent.DRL_prediction(model=trained_ddpg,
                        test_data = test_df,
                        test_env = env_trade,
                        test_obs = obs_trade)
ddpg_test_weights.to_csv('ddpg_test_weights.csv')

a2c_test_portfolio = a2c_test_daily_return.copy()
a2c_test_returns = a2c_test_daily_return.copy()

ppo_test_portfolio = ppo_test_daily_return.copy()
ppo_test_returns = ppo_test_daily_return.copy()

ddpg_test_portfolio = ddpg_test_daily_return.copy()
ddpg_test_returns = ddpg_test_daily_return.copy()

warnings.simplefilter(action='ignore', category=FutureWarning)

a2c_train_cum_returns = (1 + a2c_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
a2c_train_cum_returns = a2c_train_cum_returns['daily_return']
a2c_train_cum_returns.name = 'Portfolio: a2c Model'

ppo_train_cum_returns = (1 + ppo_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
ppo_train_cum_returns = ppo_train_cum_returns['daily_return']
ppo_train_cum_returns.name = 'Portfolio: ppo Model'

ddpg_train_cum_returns = (1 + ddpg_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
ddpg_train_cum_returns = ddpg_train_cum_returns['daily_return']
ddpg_train_cum_returns.name = 'Portfolio: ddpg Model'

date_list = list(ddpg_train_cum_returns.index)

# Plot the culmulative returns of the portfolios
fig, ax = plt.subplots(figsize=(8,4))

a2c_train_cum_returns.plot(ax=ax, color='blue', alpha=0.4)
ppo_train_cum_returns.plot(ax=ax, color='green', alpha=0.4)
ddpg_train_cum_returns.plot(ax=ax, color='purple', alpha=0.4)

plt.legend(loc="best")
plt.grid(True)
ax.set_ylabel("cummulative return")
ax.set_title("Backtest based on the data from 2019-01-01 to 2020-12-31", fontsize=14)
fig.savefig('results/back_test_on_train_data.png')


a2c_test_cum_returns = (1 + a2c_test_returns['daily_return']).cumprod()
a2c_test_cum_returns.name = 'Portfolio: a2c Model'

ppo_test_cum_returns = (1 + ppo_test_returns['daily_return']).cumprod()
ppo_test_cum_returns.name = 'Portfolio: ppo Model'

ddpg_test_cum_returns = (1 + ddpg_test_returns['daily_return']).cumprod()
ddpg_test_cum_returns.name = 'Portfolio: ddpg Model'

a2c_test_cum_returns.to_csv('a2c_test_cum_returns.csv')
ppo_test_cum_returns.to_csv('ppo_test_cum_returns.csv')
ddpg_test_cum_returns.to_csv('ddpg_test_cum_returns.csv')
# Plot the culmulative returns of the portfolios
fig, ax = plt.subplots(figsize=(8,4))

a2c_test_cum_returns.plot(ax=ax, color='blue', alpha=.4)
ppo_test_cum_returns.plot(ax=ax, color='green', alpha=.4)
ddpg_test_cum_returns.plot(ax=ax, color='purple', alpha=.4)
plt.legend(loc="best");
plt.grid(True);
ax.set_ylabel("cummulative return");
ax.set_title("Backtest based on the data from 2018-10-19 to 2020-12-30", fontsize=14);
fig.savefig('results/back_test_on_test_data.png');


# Portfolio Statistics Calculation
def portfolio_stats(portfolio_returns):
    # Pass the returns into a dataframe
    port_rets_df = pd.DataFrame(portfolio_returns)
    port_rets_df = port_rets_df.reset_index()
    port_rets_df.columns = ['date', 'daily_return']
    DRL_strat = backtest_strat(port_rets_df)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                               factor_returns=DRL_strat,
                               positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.columns = ['Statistic']
    return perf_stats_all


portfolios_returns_dict = {#'uniform_weights': uw_test_returns, #'maximum_sharpe': max_sharpe_test_returns,
                           'a2c Model': a2c_test_returns['daily_return'],
                           'ppo Model': ppo_test_returns['daily_return'],
                           'ddpg Model': ddpg_test_returns['daily_return']}

portfolios_stats = pd.DataFrame()
for i, j in portfolios_returns_dict.items():
    port_stats = portfolio_stats(j)
    portfolios_stats[i] = port_stats['Statistic']


portfolios_returns_dict = {#'uniform_weights':uw_test_returns, 'maximum_sharpe':max_sharpe_test_returns,
                          'a2c Model': a2c_test_returns['daily_return'],
                          'ppo Model': ppo_test_returns['daily_return'],
                          'ddpg Model': ddpg_test_returns['daily_return']}

portfolios_stats = pd.DataFrame()
for i,j in portfolios_returns_dict.items():
    port_stats = portfolio_stats(j)
    portfolios_stats[i] = port_stats['Statistic']

a2c_test_returns = a2c_test_returns.set_index('date')
ppo_test_returns = ppo_test_returns.set_index('date')
ddpg_test_returns = ddpg_test_returns.set_index('date')

# Combine cumulative returns
ps_cum = [a2c_test_cum_returns, ppo_test_cum_returns, ddpg_test_cum_returns]
ps = [a2c_test_returns['daily_return'], ppo_test_returns['daily_return'], ddpg_test_returns['daily_return']]
final_return = []
for p in ps_cum:
    final_return.append(p.iloc[-1])

id_ = np.argmax(final_return)
best_p = ps[id_]
best_p.name = (ps_cum[id_]).name

print("Best portfolio: ", best_p.name)
print("Final cumulative return: {:.2f} ".format(final_return[id_]))
