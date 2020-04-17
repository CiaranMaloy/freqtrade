# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class FindProfitRanges(IStrategy):
    """
    author@: Gert Wohlgemuth
    idea:
        this strategy is based on the book, 'The Simple Strategy' and can be found in detail here:
        https://www.amazon.com/Simple-Strategy-Powerful-Trading-Futures-ebook/dp/B00E66QPCG/ref=sr_1_1?ie=UTF8&qid=1525202675&sr=8-1&keywords=the+simple+strategy

        freqtrade backtesting --strategy FindProfitRanges --timerange=20200414- --export=trades
        & freqtrade plot-dataframe --strategy FindProfitRanges --timerange=20200414- --indicators2 sentiment
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    #minimal_roi = {'0': 0.19128, '15': 0.05055, '21': 0.0157, '91': 0}
    minimal_roi = {'0': 1.000, '15': 1.000, '21': 1.000, '91': 1.0}
    # keys_to_strings

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    #stoploss = -0.1302
    stoploss = -0.5



    # Optimal ticker interval for the strategy
    ticker_interval = '3m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # {'drawdown': 2.97997, 'percent': 0.67458, 'tick': 1}
        # {'drawdown': 9.80957, 'percent': 0.35411, 'tick': 26}

        # Progressive isn't something that should be considered linear
        # Sentiment = pow(sentiment, ProgressivePower)
        progressive_power = 1.5

        upperLine = 1.01
        # Percent = 1
        # tick = 26
        # drawdown = 9.8
        oneHour = 20  # 60/3

        import numpy as np

        occupied_array = np.zeros(len(dataframe))

        max_profit_endidx_array = np.zeros(len(dataframe))
        sentiment_array = np.ones(len(dataframe))
        sentiment_progressive_array = np.zeros(len(dataframe))
        DataTable = dataframe.T

        lenTable = len(dataframe)
        column_close = 4
        previousOccupiedSum=0
        while (1):
            max_profit_array = np.zeros(len(dataframe))
            for index in range(lenTable):
                if (index==1447):
                    breakHere=0
                max_idx = index
                max_profit = 1.0
                if (occupied_array[index] == 1):
                    continue
                currentClose = DataTable[index][column_close]
                for x in range(1, 1+oneHour):
                    if (index + x >= len(dataframe)):
                        # We dont have the data to predict index
                        #max_profit = 1
                        break
                    if (occupied_array[index+x] == 1):
                        break;
                    nextclose = DataTable[index + x][column_close]
                    profit = nextclose / currentClose
                    if profit > max_profit:
                        max_profit = profit
                        max_idx = index + x



                max_profit_array[index] = max_profit
                max_profit_endidx_array[index] = max_idx

            winning_idx = 1
            winning_profit = 0
            for index in range(lenTable):
                if (occupied_array[index] == 1):
                    continue
                if max_profit_array[index] > winning_profit:
                    winning_idx = index
                    winning_profit = max_profit_array[index]

            max_range_idx = int(max_profit_endidx_array[winning_idx])
            if (winning_profit > 1.005):
                sentiment_array[winning_idx] = min(winning_profit,upperLine)
                sentiment_array[max_range_idx] = 0.99
                for x in range(winning_idx, max_range_idx):
                    closedAt = DataTable[max_range_idx][column_close]
                    profit = closedAt / DataTable[x][column_close] -1
                    profit = profit/(max_profit_array[winning_idx]-1)
                    sentiment_progressive_array[x] = pow(profit, progressive_power)

            for x in range(winning_idx, max_range_idx):
                occupied_array[x] = 1




            sumOccupied = sum(occupied_array)
            print(sumOccupied)
            if (previousOccupiedSum == sumOccupied):
                break;

            previousOccupiedSum = sumOccupied

#            from sklearn.preprocessing import StandardScaler, minmax_scale, Normalizer
 #           sc = StandardScaler()
  #          X = sc.fit_transform(a.reshape(-1, 1))



        dataframe['sentiment'] = sentiment_array
        dataframe['sentiment_progressive'] = sentiment_progressive_array

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        #

        # plt.plot(dataframe['sentiment'])
        # plt.show()
        # plt.plot(a)
        # plt.show()
        # # seaborn histogram
        # sns.distplot(dataframe['sentiment'], hist=True, kde=False, color='blue',
        #              hist_kws={'edgecolor': 'black'})
        # plt.show()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['sentiment'] > 1.005)
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%

        dataframe.loc[
            (
                (dataframe['sentiment'] < 1)
            ),
            'sell'] = 1
        return dataframe
