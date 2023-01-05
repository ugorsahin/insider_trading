import os
import glob
from datetime import timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import mplfinance as mpf
from marker_map import MARKER_MAP
from special_frames import InsiderDataFrame, StockDataFrame


class StockViewer:
    table_colors = sns.color_palette(n_colors=30)

    def __init__(self, insider_path):
        csv_files = glob.glob(os.path.join(insider_path, "*.csv"))
        self.stocks = {}
        for csv_file in csv_files:
            stock_name = csv_file.split("/")[-1][:-4]
            insider_data = InsiderDataFrame(pd.read_csv(csv_file))
            insider_data.Date = pd.to_datetime(insider_data.Date)
            insider_data["Value"] = insider_data["Value ($)"].apply(
                lambda x: float(x.replace(",", "")) if isinstance(x, str) else x)
            insider_data["Share"] = insider_data["Shares"].apply(lambda x: float(
                x.replace(",", "")) if isinstance(x, str) else x)
            self.stocks[stock_name] = {
                "insider_data": insider_data, "insider_size": len(insider_data)}

    def get_insider_data(self, ticker_name):
        if (stock_dict := self.stocks.get(ticker_name)) is None:
            raise KeyError("Stock Ticker doesn't exist in database")
        return stock_dict["insider_data"]

    def get_ticker_data(self, ticker_name, start_date="2021-01-01", end_date="2023-01-01", force_download=False):
        if (stock_dict := self.stocks.get(ticker_name)) is None:
            raise KeyError("Stock Ticker doesn't exist in database")

        if (not force_download) and (stock_data := stock_dict.get("stock_data")) is not None:
            return stock_data

        stock_data = StockDataFrame(yf.download(
            ticker_name, start_date, end_date))
        self.stocks[ticker_name]["stock_data"] = stock_data
        return stock_data

    @staticmethod
    def get_moving_average(stock, rate=20):
        return stock.rolling(rate).mean()

    @staticmethod
    def get_exponential_moving_average(stock, span=20):
        return stock.ewm(span=span).mean()

    @staticmethod
    def get_bollinger_bands(stock, rate=20):
        sma = StockViewer.get_moving_average(stock, rate)
        std = stock.rolling(rate).std()
        bollinger_up = sma + std * 2
        bollinger_down = sma - std * 2
        return bollinger_up, bollinger_down

    def get_stock_insider_data(self, ticker_name, **kwargs):
        # Arguments for redownload data and week transformation
        force_download = kwargs.get("force_download") or False
        # Select relative start and end dates for the stock
        bw_delta = kwargs.get("backward_timedelta") or 120
        fw_delta = kwargs.get("forward_timedelta") or 120

        # Find the stock by ticker name
        insider_data = self.get_insider_data(ticker_name)

        # Get the stock prices for ticker
        start_date = insider_data.Date.min() - timedelta(days=bw_delta)
        end_date = insider_data.Date.max() + timedelta(days=fw_delta)
        stock_data = self.get_ticker_data(
            ticker_name, start_date=start_date, end_date=end_date, force_download=force_download)

        # Get data and transform it into plottable version
        insider_data.prepare_insider_data(stock_data)

        return stock_data, insider_data

    def get_indicators(self, stock_data, **kwargs):
        bollinger = kwargs.get("bollinger") or False
        ema = kwargs.get("ema") or False
        ma = kwargs.get("ma") or False

        indicators = {}
        if bollinger:
            b_upper, b_lower = self.get_bollinger_bands(stock_data['Close'])
            indicators.update({
                "b_upper": b_upper,
                "b_lower": b_lower
            })
        if ema:
            ema_values = kwargs.get("ema_values") or [20]
            for ema_span in ema_values:
                ema_line = self.get_exponential_moving_average(
                    stock_data['Close'], ema_span)
                indicators[f"ema_{ema_span}"] = ema_line
        if ma:
            ma_values = kwargs.get("ma_values") or [20]
            for ma_span in ma_values:
                ma_line = self.get_moving_average(stock_data['Close'], ma_span)
                indicators[f"ema_{ma_span}"] = ma_line
        return indicators

    def plot_stock_data(self, ticker_name=None, stock_data=None, insider_data=None, **kwargs):
        volume = kwargs.get("volume") or False
        style = kwargs.get("style") or "classic"
        marker_pwr = kwargs.get("marker_pwr") or 0.35
        figsize = kwargs.get("figsize") or (40, 10)
        weekly = kwargs.get("weekly") or False

        if ticker_name is None and (stock_data is None or insider_data is None):
            raise RuntimeError(
                'You should either name the ticker or provide both stock and insider data')

        if ticker_name:
            stock_data, insider_data = self.get_stock_insider_data(
                ticker_name, **kwargs)

        y_max = np.maximum(stock_data.High.max(),
                           insider_data.Cost.max()) * 1.05
        y_min = np.minimum(stock_data.Low.min(),
                           insider_data.Cost.min()) / 1.05

        # if weekly is True, the data is transformed into weekly view
        stock_data = stock_data.get_view(weekly=weekly)
        insider_data = insider_data.get_by_person(weekly=weekly)

        plots = []
        # For each person who have a transaction, we will have a different color
        # The size of transaction will be relative to total value of transaction
        # The marker will be different w.r.t. transaction type
        for idx, _person in enumerate(insider_data):
            for t_name, dframe in _person.items():
                person_scatter = mpf.make_addplot(
                    dframe["Cost"], type="scatter",
                    color=self.table_colors[idx],
                    markersize=dframe["Value"] ** marker_pwr,
                    marker=MARKER_MAP[t_name],
                    ylim=(y_min, y_max))
                plots.append(person_scatter)

        # Add the indicators to figure
        indicators = [mpf.make_addplot(indicator)
                      for indicator in self.get_indicators(stock_data, **kwargs).values()]

        if (external_plots := kwargs.get('external_plots')) is not None:
            plots.extend(external_plots)
        plots.extend(indicators)

        def timify(x, t): return x.index[t].strftime("%m-%d-%Y")

        title = f"""{ticker_name or ''} prices - {'weekly' if weekly else 'daily'}
        between {timify(stock_data, 0)} - {timify(stock_data, -1)}"""

        _, axis = mpf.plot(
            stock_data, type="candle", style=style, volume=volume,
            figsize=figsize, addplot=plots, returnfig=True, title=title, ylim=(y_min, y_max))

        # If requested view is weekly, every week will have a minor gridline
        # otherwise every 7 days will be.
        tick_step = 1 + 6 * (not weekly)
        minor_ticks = np.arange(0, len(stock_data)+1, tick_step)
        axis[0].set_xticks(minor_ticks)


def assign_simple_patterns(candles):
    bottom = candles.Low.rolling(5, min_periods=1).min() == candles.Low
    top = candles.High.rolling(5, min_periods=1).max() == candles.High
    up = candles.Close.rolling(5, min_periods=1).max() == candles.Close
    down = candles.Close.rolling(5, min_periods=1).min() == candles.Close

    count = 0
    cons_up = [count := count+1 if t else 0 for t in up]
    count = 0
    cons_down = [count := count+1 if t else 0 for t in down]

    candle_len = candles.High - candles.Low
    high_open = candles.High - candles.Open
    open_low = candles.Open - candles.Low
    high_close = candles.High - candles.Close
    close_low = candles.Close - candles.Low
    close_open = candles.Close - candles.Open

    positive = close_open >= 0
    negative = close_open < 0

    body = close_open * positive + close_open * -1 * negative
    upper_tail = high_close * positive + high_open * negative
    lower_tail = open_low * positive + close_low * negative

    ratio = body / candle_len
    marubozu = ratio > 0.99
    big_body = ratio >= 0.6
    doji = ratio < 0.1

    black_body = np.logical_and.reduce(
        (negative, big_body, np.logical_not(marubozu)))
    white_body = np.logical_and.reduce(
        (positive, big_body, np.logical_not(marubozu)))

    dragonfly_doji = np.logical_and(doji, upper_tail/lower_tail < 0.1)
    gravestone_doji = np.logical_and(doji, lower_tail/upper_tail < 0.1)

    hammer = np.logical_and(
        upper_tail / candle_len < 0.05,
        lower_tail / candle_len > 0.5
    )
    hanging_man = np.logical_and(top, hammer)
    hammer = np.logical_and(bottom, hammer)

    inv_hammer = np.logical_and(
        upper_tail / candle_len > 0.5,
        lower_tail / candle_len < 0.05
    )
    shooting_star = np.logical_and(top, inv_hammer)
    inv_hammer = np.logical_and(bottom, inv_hammer)

    spinning = np.logical_and.reduce((
        ratio < 0.5,
        upper_tail/candle_len > 0.1,
        lower_tail/candle_len > 0.1,
        np.logical_not(doji)
    ))

    spinning_white = np.logical_and(positive, spinning)
    spinning_black = np.logical_and(negative, spinning)

    unidentified = np.logical_not(
        np.logical_or.reduce((
            marubozu, doji, black_body, white_body, spinning_white, spinning_black,
            dragonfly_doji, gravestone_doji, hanging_man, hammer, shooting_star, inv_hammer)
        ))

    candles = candles.assign(
        close_open=close_open,
        upper_tail=upper_tail,
        lower_tail=lower_tail,
        body=body,
        candle_len=candle_len,
        upward=up,
        downward=down,
        consecutive_up=cons_up,
        consecutive_down=cons_down,
        bottom=bottom,
        top=top,
        marubozu=marubozu,
        big_body=big_body,
        doji=doji,
        black_body=black_body,
        white_body=white_body,
        dragonfly_doji=dragonfly_doji,
        gravestone_doji=gravestone_doji,
        hanging_man=hanging_man,
        hammer=hammer,
        shooting_star=shooting_star,
        inv_hammer=inv_hammer,
        spinning_black=spinning_black,
        spinning_white=spinning_white,
        unidentified=unidentified
    )

    return candles


CANDLE_PATTERNS = [
    'marubozu', 'doji', 'black_body', 'white_body',
    'dragonfly_doji', 'gravestone_doji', 'hanging_man', 'hammer',
    'shooting_star', 'inv_hammer', 'spinning_white', 'spinning_black', 'unidentified']


def summarize_candles(stock_pattern):
    sum_arr = stock_pattern[CANDLE_PATTERNS].sum()
    return sum_arr


def summarize_occurences(stock_pattern, insider_data):
    sum_arr = np.zeros(len(CANDLE_PATTERNS), dtype=int)

    for _person in insider_data:
        if (sales := _person.get("Sale")) is None:
            continue
        sum_arr += [
            np.logical_and(~sales.Insider.isna(), stock_pattern[pat]).sum()
            for pat in CANDLE_PATTERNS
        ]

    return dict(zip(CANDLE_PATTERNS, sum_arr))
