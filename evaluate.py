import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
#0: wealth, 1: reward, 2: action, 3: weight, 4: price
def plot_result(num, istrain):
    with open('config.json') as f:
        config = json.load(f)
    tickers = config["session"]["tickers"]

    check_epoch = [0, 25, 50, 75, 100]
    #plot train
    if istrain:
        for epoch in check_epoch:
            file_name = "result/PG/" + num + '/' + "PG-" + str(epoch) + '.pkl'
            with open(file_name, 'rb') as pk:
                result = pickle.load(pk)
            plt.plot(range(len(result)), result.iloc[:, 0], label="epoch " + str(epoch))
        plt.legend()
        plt.show()
        for epoch in check_epoch[1:]:
            file_name = "result/PG/" + num + '/' + "PG-" + str(epoch) + '.pkl'
            with open(file_name, 'rb') as pk:
                result = pickle.load(pk)
            weights = pd.DataFrame(result.iloc[:, 2].to_list(), columns=["cash"] + tickers)
            weights = weights.loc[:, (weights != 0).any(axis=0)]
            weights.plot(kind='bar', stacked=True, title="weight distribution in epoch " + str(epoch), xticks=[], xlabel="Time")
            plt.show()
    #plot test
    else:
        #plot portfolio values over time
        for agent in ["PG", "Equal_Weight"]:
            file_name = "result/PG/" + num + '/' + agent + '-test.pkl'
            with open(file_name, 'rb') as pk:
                result = pickle.load(pk)
            plt.plot(range(len(result)), result.iloc[:, 0], label=agent)
        plt.legend()
        plt.savefig("backtest_wealth.png")
        plt.show()
        #plot weight distribution
        for agent in ["PG", "Equal_Weight"]:
            file_name = "result/PG/" + num + '/' + agent + '-test.pkl'
            with open(file_name, 'rb') as pk:
                result = pickle.load(pk)
            weights = pd.DataFrame(result.iloc[:, 2].to_list(), columns=["cash"] + tickers)
            weights = weights.loc[:, (weights != 0).any(axis=0)]

            if agent == "PG":
                weights.plot(kind='bar', stacked=True, title="weight distribution using "+agent, xticks=[], xlabel="Time")
            else:
                weights.plot(kind='bar', stacked=True, title="weight distribution using "+agent, xticks=[], xlabel="Time", legend=False)
            plt.savefig('weights_'+agent+'.png')
            plt.show()


def compare_window():
    agent = "PG"
    file_name5 = "result/PG/7/PG-test.pkl"
    file_name10 = "result/PG/2/PG-test.pkl"
    file_name20 = "result/PG/6/PG-test.pkl"
    file_name50 = "result/PG/5/PG-test.pkl"
    with open(file_name5, 'rb') as pk:
        result = pickle.load(pk)
    plt.plot(range(len(result)), result.iloc[:, 0], label="window = 5")

    with open(file_name10, 'rb') as pk:
        result = pickle.load(pk)
    plt.plot(range(10, 10+len(result)), result.iloc[:, 0], label="window = 10", color="red")

    with open(file_name20, 'rb') as pk:
        result = pickle.load(pk)
    plt.plot(range(30, len(result)+30), result.iloc[:, 0], label="window = 20")

    with open(file_name50, 'rb') as pk:
        result = pickle.load(pk)
    plt.plot(range(90, len(result)+90), result.iloc[:, 0], label="window = 50")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # compare_window()
    plot_result("3", True)