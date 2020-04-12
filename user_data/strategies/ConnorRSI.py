# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import talib
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ConnorRSI(IStrategy):
    """
    author@: Gert Wohlgemuth
    idea:
        this strategy is based on the book, 'The Simple Strategy' and can be found in detail here:
        https://www.amazon.com/Simple-Strategy-Powerful-Trading-Futures-ebook/dp/B00E66QPCG/ref=sr_1_1?ie=UTF8&qid=1525202675&sr=8-1&keywords=the+simple+strategy

        Best result:

    86/100:     27 trades. Avg profit   0.38%. Total profit  103.81731543 USDT (  10.37?%). Avg duration  67.9 min. Objective: 1.90737

Best result:

*   15/100:     17 trades. Avg profit   0.33%. Total profit  55.48218535 USDT (   5.54?%). Avg duration  71.6 min. Objective: 1.93118

Buy hyperspace params:
{'connorMin': 5}
Sell hyperspace params:
{'connorMax': 88}
ROI table:
{0: 0.18528, 15: 0.03847, 37: 0.0272, 62: 0}
Stoploss: -0.26241

    """


    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = \
        {
            "0": 0.18528,
            "15": 0.03847,
            "37": 0.0272,
            "62": 0
        }



    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.26241

    # Optimal ticker interval for the strategy
    ticker_interval = '3m'

    connorMin = 5
    connorMax = 88

    # ConnorsRSI(3, 2, 100) = [RSI(Close, 3) + RSI(Streak, 2) + PercentRank(100)] / 3
    # ConnorsRSI(3,2,100) = [ RSI(Close,3) + RSI(Streak,2) + PercentRank(percentMove,100) ] / 3
    # a. Short term Relative Strength, i.e., RSI(3).
    # b. Counting consecutive up and down days (streaks) and “normalizing” the data using RSI(streak,2). The result is a bounded, 0-100  indicator.
    # c. Magnitude of the move (percentage-wise) in relation to previous moves. This is measured using the percentRank() function.

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=3)
        close = dataframe['close']

        dataframe["closenext"] = close.shift(1)
        dataframe.describe()

        #        import numpy as np
        greater = dataframe["closenext"] > dataframe["close"]
        dataframe["Streak"] = (greater.groupby((greater != greater.shift()).cumsum()).cumcount(ascending=False))
        #        prices = np.array(dataframe["Streak"].values, dtype=float)

        dataframe["Streak_RSI"] = talib.RSI(dataframe["Streak"], timeperiod=2)
        df = dataframe["Streak_RSI"]
        dataframe["Streak_RSI_n"] = (df - df.min()) / (df.max() - df.min()) * 100
        dataframe["ROC"] = ta.ROC(dataframe)

        # [RSI(Close, 3) + RSI(Streak, 2) + PercentRank(percentMove, 100)] / 3
        dataframe["ConnorsRSI"] = (dataframe['rsi'] + dataframe["Streak_RSI"] + dataframe["ROC"]) / 3

        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        #
        # # RSI
        # #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        #
        # # required for graphing
        # bollinger = qtpylib.bollinger_bands(dataframe['close'], window=12, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_upperband'] = bollinger['upper']
        # dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['ConnorsRSI'] < self.connorMin)  # over 0
                    # & (dataframe['macd'] > dataframe['macdsignal'])  # over signal
                    # & (dataframe['bb_upperband'] > dataframe['bb_upperband'].shift(1))  # pointed up
                    # & (dataframe['rsi'] > 70)  # optional filter, need to investigate
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%
        dataframe.loc[
            (
                (dataframe['ConnorsRSI'] > self.connorMax)
            ),
            'sell'] = 1
        return dataframe
