# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class FindBuySell(IStrategy):
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
    #keys_to_strings

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1302

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #{'drawdown': 2.97997, 'percent': 0.67458, 'tick': 1}
        #{'drawdown': 9.80957, 'percent': 0.35411, 'tick': 26}

        Percent = 0.35
        tick = 26
        drawdown = 9.8
        import numpy as np
        a = np.array([])
        for index, row in dataframe.iterrows():
            for x in range(1, 1+tick):
                buy = False
                if(index + x >= len(dataframe)):
                    break
                nextclose = dataframe.iloc[index + x]["close"]
                profit =  nextclose > row["close"] * (1 + Percent / 100)
                #print(row["date"], nextclose / row["close"])
                draw =  nextclose < row["close"] * (1 - drawdown / 100)
                if draw:
                    buy = False
                    break
                if (profit):
                    buy = True
                    break
            a = np.concatenate((a, np.array(buy)), axis=None)

        dataframe['ShouldBuy'] = a

        a = np.array([])
        for index, row in dataframe.iterrows():
            sell = False
            if row['ShouldBuy'] == 1 or index < tick:
                sell  = False
            else:
                for x in range(1,1+tick):
                    prevClose = dataframe.iloc[index - x]["close"]
                    profit = prevClose * (1 + Percent / 100) < row["close"]
                    if(profit):
                        sell = True;
                        break
            a = np.concatenate((a, np.array(sell)), axis=None)

        dataframe['ShouldSell'] = a
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['ShouldBuy'] > 0)
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%

        dataframe.loc[
            (
                (dataframe['ShouldSell']>0)
            ),
            'sell'] = 1
        return dataframe