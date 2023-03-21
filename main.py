from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
plt.style.use('ggplot')
from agents.OU import OrnsteinUhlenbeckActionNoise
from agents.pg import PG

import os
import seaborn as sns

sns.set_style("darkgrid")

eps = 10e-8
epochs = 0
M = 0
PATH_prefix = ''


class StockTrader():
    def __init__(self):
        self.reset()

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss = 0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def update_summary(self, loss, r, q_value, actor_loss, w, p):
        self.loss += loss
        self.actor_loss += actor_loss
        self.total_reward += r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([[round(w0, 2) for w0 in w[0]]])
        self.p_history.extend([[round(p0, 3) for p0 in p]])


    def write(self, epoch, agent):
        global PATH_prefix
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat([wealth_history, r_history, w_history, p_history], axis=1)
        file_name = PATH_prefix + agent + '-' + str(epoch) + ".pkl"
        with open(file_name, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def print_result(self, epoch, agent, noise_flag):
        self.total_reward = math.exp(self.total_reward) * 100
        print('*-----Episode: {:d}, Reward:{:.6f}%-----*'.format(epoch, self.total_reward))
        agent.write_summary(self.total_reward)
        agent.save_model()

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self, a, ratio):
        a = np.clip(a + self.noise() * ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a


def parse_info(info):
    return info['reward'], info['continue'], info['next state'], info['weight vector'], info['price'], info['risk']


def traversal(stocktrader, agent, env, epoch, noise_flag, framework, method, trainable):
    info = env.step(None, None, noise_flag)
    r, contin, s, w1, p, risk = parse_info(info)
    contin = 1
    t = 0

    while contin:
        w2 = agent.predict(s, w1)

        env_info = env.step(w1, w2, noise_flag)
        r, contin, s_next, w1, p, risk = parse_info(env_info)

        if framework == 'PG':
            agent.save_transition(s, p, w2, w1)

        loss, q_value, actor_loss = 0, 0, 0

        if framework == 'PG':
            if not contin and trainable == "True":
                agent.train()

        stocktrader.update_summary(loss, r, q_value, actor_loss, w2, p)
        s = s_next
        t = t + 1


def maxdrawdown(arr):
    i = np.argmax((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr))  # end of the period
    j = np.argmax(arr[:i])  # start of period
    return (1 - arr[i] / arr[j])


def backtest(agent, env):
    global PATH_prefix
    print("starting to backtest......")
    from agents.EqWeight import EqWeights

    agents = []
    agents.extend(agent)
    agents.append(EqWeights())

    labels = ['PG', 'Equal_Weight', ]

    wealths_result = []
    rs_result = []
    for i, agent in enumerate(agents):
        stocktrader = StockTrader()
        info = env.step(None, None, 'False')
        r, contin, s, w1, p, risk = parse_info(info)
        contin = 1
        wealth = 10000
        wealths = [wealth]
        rs = [1]
        while contin:
            w2 = agent.predict(s, w1)
            env_info = env.step(w1, w2, 'False')
            r, contin, s_next, w1, p, risk = parse_info(env_info)
            wealth = wealth * math.exp(r)
            rs.append(math.exp(r) - 1)
            wealths.append(wealth)
            s = s_next
            stocktrader.update_summary(0, r, 0, 0, w2, p)

        stocktrader.write("test", labels[i])
        print('finish one agent')
        wealths_result.append(wealths)
        rs_result.append(rs)


    print("Asset    Avg Daily Return     Sharpe Ratio    Max Drawdown")
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        plt.plot(wealths_result[i], label=labels[i])
        mrr = float(np.mean(rs_result[i]) * 100)
        sharpe = float(np.mean(rs_result[i]) / np.std(rs_result[i]) * np.sqrt(252))
        maxdrawdown = float(max(1 - min(wealths_result[i]) / np.maximum.accumulate(wealths_result[i])))
        print(labels[i], '   ', round(mrr, 3), '%', '   ', round(sharpe, 3), '  ', round(maxdrawdown, 3))
    plt.legend()
    plt.savefig(PATH_prefix + 'backtest.png')
    plt.show()


def parse_config(config, mode):
    num_stock = config["session"]["num_stock"]
    train_start_date = config["session"]["train_start_date"]
    train_end_date = config["session"]["train_end_date"]
    test_start_date = config["session"]["test_start_date"]
    test_end_date = config["session"]["test_end_date"]
    tickers = config["session"]["tickers"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    noise_flag, record_flag, plot_flag = config["session"]["noise_flag"], config["session"]["record_flag"], \
                                         config["session"]["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable = config["session"]['reload_flag'], config["session"]['trainable']
    method = config["session"]['method']
    normalize_method = config["session"]["normalize_method"]
    transaction_fee = config["session"]["transaction_cost"]
    global epochs
    epochs = int(config["session"]["epochs"])

    if mode == 'test':
        record_flag = True
        noise_flag = 'False'
        plot_flag = 'True'
        reload_flag = 'True'
        trainable = 'False'
        method = 'model_free'
        print("*--------------------Test Status-------------------*")
        print("Date from", test_start_date, ' to ', test_end_date)

    if mode =="train":
        print("*--------------------Training Status-------------------*")
        print("Date from", train_start_date, ' to ', train_end_date)

    print('Features:', features)
    print("Agent:Noise(", noise_flag, ')---Record(', record_flag, ')---Plot(', plot_flag, ')')
    print("Market Type:", market)
    print("Predictor:", predictor, "  Framework:", framework, "  Window_length:", window_length)
    print("Epochs:", epochs)
    print("Trainable:", trainable)
    print("Reloaded Model:", reload_flag)
    print("Method", method)
    print("Noise_flag", noise_flag)
    print("Record_flag", record_flag)
    print("Plot_flag", plot_flag)
    print("Normalize_method", normalize_method)
    print("Transaction_cost_pct", transaction_fee)
    return num_stock, tickers, train_start_date, train_end_date, test_start_date, test_end_date, features, agent_config, market, \
           predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method, normalize_method, transaction_fee


def session(config, args):
    global PATH_prefix
    from data.environment import Environment
    num_stock, tickers, train_start_date, train_end_date, test_start_date, test_end_date, features, agent_config, market, \
        predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method, normalize_method, transaction_fee = parse_config(
        config, args)
    env = Environment(transaction_fee)

    global M
    M = num_stock + 1

    stocktrader = StockTrader()
    PATH_prefix = "result/PG/" + str(args['num']) + '/'

    if args['mode'] == 'train':
        if not os.path.exists(PATH_prefix):
            os.makedirs(PATH_prefix)
        env.get_data(train_start_date, train_end_date, features, window_length, market, tickers, normalize_method)

        for noise_flag in ['True']:  # ['False','True'] to train agents with noise and without noise in assets prices
            framework = 'PG'
            print("*-----------------Loading PG Agent---------------------*")
            agent = PG(len(tickers) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,
                       trainable, noise_flag, args['num'])

            print("Training with {:d} epoches".format(epochs))
            for epoch in range(epochs):
                print("Now we are at epoch", epoch)
                traversal(stocktrader, agent, env, epoch, noise_flag, framework, method, trainable)

                if record_flag == 'True':
                    if epoch % 25 == 0:
                        stocktrader.write(epoch, framework)
                    if epoch == epochs - 1:
                        stocktrader.write(epoch+1, framework)

                agent.reset_buffer()
                stocktrader.print_result(epoch, agent, noise_flag)
                stocktrader.reset()

            agent.close()
            del agent

    elif args['mode'] == 'test':

        env.get_data(test_start_date, test_end_date, features, window_length, market, tickers, normalize_method)
        backtest([PG(len(tickers) + 1, int(window_length), len(features), '-'.join(agent_config),
                     load_weights=True, trainable=False, type="True", number=args['num'])],
                 env)


def build_parser():
    parser = ArgumentParser(
        description='Provide arguments for training models in Portfolio Management')
    parser.add_argument("--mode", choices=['train', 'test'])
    parser.add_argument("--num", type=int)
    return parser


def main():
    # parser = build_parser()
    # args=vars(parser.parse_args())
    args = {'mode': "train", "num": 4} #define mode: {train, test} and session number
    with open('config.json') as f:
        config = json.load(f)
        session(config, args)


if __name__ == "__main__":
    main()
