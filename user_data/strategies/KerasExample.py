# --- Do not remove these libs ---
import talib
import numpy as np
import tensorflow_core as tf
from tensorflow_core import metrics
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


def SetGPUMemoryGrowth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


class KerasExample(IStrategy):
    """
    author@: Gert Wohlgemuth
    idea:
        this strategy is based on the book, 'The Simple Strategy' and can be found in detail here:
        https://www.amazon.com/Simple-Strategy-Powerful-Trading-Futures-ebook/dp/B00E66QPCG/ref=sr_1_1?ie=UTF8&qid=1525202675&sr=8-1&keywords=the+simple+strategy
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.01
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.25

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        SetGPUMemoryGrowth()

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
        dataframe["ConnorsRSI"] = (dataframe['rsi'] + dataframe["Streak_RSI_n"] + dataframe["ROC"]) / 3


        #todo Faire un should buy dans le sens du monde et un sell
        dataframe["ShouldBuy"] = dataframe["close"].shift(-5) > dataframe["close"]
        dataframe["ShouldBuy"] = dataframe["ShouldBuy"].astype(int)

        #remove first with NAN
        Newdataframe = dataframe.iloc[40:,:]

        #remove column we dont want to train on
        X = Newdataframe.drop(["date", "ShouldBuy"],axis=1).values

        #keep ShouldBuy as Label
        y = Newdataframe["ShouldBuy"].values


        #X = np.expand_dims(X,1)
        y = np.expand_dims(y,1)

        # Normalizing the data
        #todo check if this is done per column and maybe normalize for days when using LSTM
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)

        #encoding buy
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y).toarray()

        #split test/train/val buy i'll be using dates most likely
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)


        #todo create a model that make sense
        model = Sequential()
        model.add(Dense(10, input_dim=12, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation="softmax"))

        #todo Add tensorboard callback
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[metrics.mae, metrics.binary_accuracy, metrics.binary_crossentropy])

        history = model.fit(X_train, y_train, epochs=10, batch_size=1024)

        #keeping for tf reference
        # test = tf.data.Dataset.from_tensor_slices((df['open'].head(1).values, df['open'].head(1).values))
        # model.predict(test)

        _, mae,accuracy,_= model.evaluate(X_test,y_test)
        print('Accuracy: %.2f' % (accuracy * 100))
        model.evaluate(X_test, y_test)
        x1 = dataframe.drop(["date", "ShouldBuy"], axis=1).values
        a = model.predict(x1)
        dataframe["ShouldBuy"] = a

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['ShouldBuy'] > 0.9)  # over 0
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # different strategy used for sell points, due to be able to duplicate it to 100%
        dataframe.loc[
            (
                (dataframe['ShouldBuy'].shift(10) == True)
            ),
            'sell'] = 1
        return dataframe