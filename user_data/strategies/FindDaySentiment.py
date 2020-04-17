# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class FindDaySentiment(IStrategy):
    """
    author@: Gert Wohlgemuth
    idea:
        this strategy is based on the book, 'The Simple Strategy' and can be found in detail here:
        https://www.amazon.com/Simple-Strategy-Powerful-Trading-Futures-ebook/dp/B00E66QPCG/ref=sr_1_1?ie=UTF8&qid=1525202675&sr=8-1&keywords=the+simple+strategy
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {'0': 0.19128, '15': 0.05055, '21': 0.0157, '91': 0}
    # keys_to_strings

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1302

    # Optimal ticker interval for the strategy
    ticker_interval = '3m'

    # params:
    sentimentBuy = 1.00542
    sentimentSell = 1.00357

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        upperLine = 1.01
        # Percent = 1
        # tick = 26
        # drawdown = 9.8
        oneHour = 20  # 60/3

        import numpy as np
        a = np.zeros(len(dataframe))
        for index, row in dataframe.iterrows():
            max_profit = 0
            for x in range(1, 1+oneHour):
                if (index + x >= len(dataframe)):
                    # We dont have the data to predict index
                    max_profit = 1
                    break
                nextclose = dataframe.iloc[index + x]["close"]
                profit = nextclose / (row["close"])
                if profit > max_profit:
                    max_profit = profit


            a[index] = min(max_profit,upperLine)

            from sklearn.preprocessing import StandardScaler, minmax_scale, Normalizer
            sc = StandardScaler()
            X = sc.fit_transform(a.reshape(-1, 1))

        dataframe['sentiment'] = a

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
                    (dataframe['sentiment'] > self.sentimentBuy)
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%

        dataframe.loc[
            (
                (dataframe['sentiment'] < self.sentimentSell)
            ),
            'sell'] = 1
        return dataframe
