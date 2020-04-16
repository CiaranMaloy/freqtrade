# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib

from filelock import FileLock, Timeout

from user_data.strategies.ConnorRSI import ConnorRSI


class BuySell_opt(IHyperOpt):
    """
    This is a Hyperopt template to get you started.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Add any lib you need to build your hyperopt.

    You must keep:
    - The prototypes for the methods: populate_indicators, indicator_space, buy_strategy_generator.

    The methods roi_space, generate_roi_table and stoploss_space are not required
    and are provided by default.
    However, you may override them if you need 'roi' and 'stoploss' spaces that
    differ from the defaults offered by Freqtrade.
    Sample implementation of these methods will be copied to `user_data/hyperopts` when
    creating the user-data directory using `freqtrade create-userdir --userdir user_data`,
    or is available online under the following URL:
    https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py.
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        import uuid

        id = uuid.uuid1()
        run_directory = "C:\\disciples_" + str(id)

        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            Percent = params["percent"]
            tick =  params["tick"]
            drawdown =  params["drawdown"]

            import numpy as np
            a = np.array([])
            buy = False
            for index, row in dataframe.iterrows():
                for x in range(1, 1+tick):
                    buy = False
                    if (index + x >= len(dataframe)):
                        break
                    nextclose = dataframe.iloc[index + x]["close"]
                    profit = nextclose > row["close"] * (1 + Percent / 100)
                    # print(row["date"], nextclose / row["close"])
                    draw = nextclose < row["close"] * (1 - drawdown / 100)
                    if draw:
                        buy = False
                        break
                    if (profit):
                        buy = True
                        break
                a = np.concatenate((a, np.array(buy)), axis=None)

            dataframe['ShouldBuy'] = a

            Percent = params["percent"]
            tick =  params["tick"]
            drawdown =  params["drawdown"]

            Sell = False
            a = np.array([])
            for index, row in dataframe.iterrows():
                sell = False
                if row['ShouldBuy'] == 1 or index < tick:
                    sell = False
                else:
                    for x in range(1, 1+tick):
                        prevClose = dataframe.iloc[index - x]["close"]
                        profit = prevClose * (1 + Percent / 100) < row["close"]
                        if (profit):
                            sell = True;
                            break
                a = np.concatenate((a, np.array(sell)), axis=None)

            dataframe['ShouldSell'] = a


            dataframe.loc[
                (
                    (
                        (dataframe['ShouldBuy'] > 0)
                    )
                ),
                'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Real(0, 10, name='percent'),
            Integer(1, 50, name='tick'),
            Real(0, 10, name='drawdown')

        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [

        ]


    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by Hyperopt.
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:



            dataframe.loc[
                (
                    (dataframe['ShouldSell'] > 0)
                ),
                'sell'] = 1
            return dataframe

        return populate_sell_trend


