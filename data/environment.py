import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
import random

eps = 10e-8


def fill_zeros(x):
    return '0' * (6 - len(x)) + x

def global_maxminscaler(subndarray):
    """
    normalize together
    """
    v = subndarray.max() - subndarray.min()
    if v == 0:
        return np.ones(subndarray.shape) * 0.5
    return (subndarray - subndarray.min()) / v

def local_maxminscaler(unindarray):
    """
    normalize separately
    """
    v = unindarray.max(axis=0) - unindarray.min(axis=0)
    unindarray[:, [v == 0][0]] = 0.5
    np.divide(unindarray - unindarray.min(axis=0), v, out=unindarray, where=v != 0)
    return unindarray

class Environment:
    def __init__(self, trans_cost):
        self.cost = trans_cost #set transaction cost here

    def get_data(self, start_time, end_time, features, window_length, market, tickers, normalize):
        self.tickers = tickers

        self.data = pd.read_csv(r'./data/' + market + '.csv', index_col=0, parse_dates=True, dtype=object)
        self.data[features] = self.data[features].astype(float)
        data = self.data[(self.data.index >= start_time) & (self.data.index <= end_time)]

        # Initialize parameters
        self.M = len(tickers) + 1
        self.N = len(features)
        self.L = int(window_length)

        price_features = list({"open", "close", "high", "low"}.intersection(set(features)))
        self.states = []
        self.price_history = []
        t = self.L
        length = len(np.unique(data.index.values))
        # generate data for each asset
        if normalize == "baseprice":
            asset_dict = dict()
            for asset in tickers:
                asset_data = data[data["tic"] == asset]  #get single asset data
                base_price = asset_data['open'].iloc[0] #use open price on the first day as base price
                base_vol = asset_data["volume"].iloc[0]
                for ftr in price_features:
                    asset_data[ftr] = asset_data[ftr] / base_price
                asset_data["volume"] = asset_data["volume"] / base_vol
                asset_dict[str(asset)] = asset_data

            # generate tensor
            while t < length:
                V = dict()
                for ftr in features:
                    V[ftr] = np.ones(self.L)
                y = np.ones(1)
                state = []

                for asset in tickers:
                    asset_data = asset_dict[str(asset)].reset_index()
                    for ftr in features:
                        V[ftr] = np.vstack((V[ftr], asset_data.loc[t - self.L: t - 1, ftr]))
                    y = np.vstack((y, asset_data.loc[t, 'close'] / asset_data.loc[t - 1, 'close']))
                for k, v in V.items():
                    state.append(v)

                state = np.stack(state, axis=1)
                state = state.reshape(1, self.M, self.L, self.N)
                self.states.append(state)
                self.price_history.append(y)
                t = t + 1

        else:
            while t < length:
                V = dict()
                for ftr in features:
                    V[ftr] = np.ones(self.L)
                y = np.ones(1)
                state = []
                for asset in tickers:
                    asset_data = data[data["tic"] == asset].reset_index()
                    #normalize price features together
                    price_norm = global_maxminscaler(asset_data.loc[t - self.L : t-1, price_features].values)
                    price_norm = pd.DataFrame(price_norm, columns=price_features)
                    for ftr in price_features:
                        V[ftr] = np.vstack((V[ftr], price_norm.loc[:, ftr]))
                    #normalize each other individual feature separately. here we only have volume
                    V["volume"] = np.vstack((V["volume"], local_maxminscaler(asset_data.loc[t - self.L : t-1, "volume"].values)))
                    y = np.vstack((y, asset_data.loc[t, 'close'] / asset_data.loc[t - 1, 'close']))
                for k, v in V.items():
                    state.append(v)

                state = np.stack(state, axis=1)
                state = state.reshape(1, self.M, self.L, self.N)
                self.states.append(state)
                self.price_history.append(y)
                t = t + 1
        self.reset()

    def step(self, w1, w2, noise):
        if self.FLAG:
            not_terminal = 1
            price = self.price_history[self.t]
            if noise == 'True':
                price = price + np.stack(np.random.normal(0, 0.002, (1, len(price))), axis=1)
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum() #transaction cost

            # std = self.states[self.t - 1][0].std(axis=0, ddof=0)
            # w2_std = (w2[0]* std).sum()

            # #adding risk
            # gamma=0.00
            # risk=gamma*w2_std

            risk = 0
            r = (np.dot(w2, price)[0] - mu)[0]

            reward = np.log(r + eps)

            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == len(self.states):
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward, 'continue': not_terminal, 'next state': self.states[self.t],
                    'weight vector': w2, 'price': price, 'risk': risk}
            return info
        else:
            info = {'reward': 0, 'continue': 1, 'next state': self.states[self.t],
                    'weight vector': np.array([[1] + [0 for i in range(self.M - 1)]]),
                    'price': self.price_history[self.t], 'risk': 0}

            self.FLAG = True
            return info

    def reset(self):
        self.t = self.L + 1
        self.FLAG = False

    def get_codes(self):
        return self.tickers
