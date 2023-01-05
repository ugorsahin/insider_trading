import pandas as pd
from marker_map import MARKER_MAP


class StockDataFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return InsiderDataFrame

    def weekly(self):
        data = self.copy()
        data['week'] = data.index.to_period('W')
        data = data.groupby(by='week').agg(
            High=('High', 'max'),
            Low=('Low', 'min'),
            Open=('Open', 'first'),
            Close=('Close', 'last'),
            Volume=('Volume', 'sum')
        )
        data.index = data.index.to_timestamp()
        return data

    def get_view(self, weekly=False):
        if weekly:
            return self.weekly()
        return self


class InsiderDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.by_person = None

    @property
    def _constructor(self):
        return InsiderDataFrame

    def split_by_transaction(self, data_index, person):
        dfs = {}
        tr_type = person.Transaction
        for name in MARKER_MAP:
            mdf = person[tr_type == name]
            matching_keys = mdf.index.intersection(data_index)
            mdf = mdf.loc[matching_keys]
            if len(mdf) > 0:
                empty_df = pd.DataFrame(index=data_index)
                empty_df = empty_df.join(mdf)
                dfs[name] = empty_df
        return dfs

    def prepare_insider_data(
            self, stock_data: pd.DataFrame = None, trans_split: bool = True):
        if stock_data is not None:
            data_index = stock_data.index
        else:
            start = self.Date.min()
            end = self.Date.max()
            data_index = pd.date_range(start, end)

        by_person = []
        for _, _df in self.groupby('Insider Trading'):
            _person = _df.groupby(['Date', 'Transaction']).agg(
                Insider=('Insider Trading', 'first'),
                Relationship=('Relationship', 'first'),
                Value=('Value', 'sum'),
                Share=('Share', 'sum')
            )
            _person['Cost'] = _person['Value'] / _person['Share']

            if trans_split:
                _person = _person.reset_index().set_index('Date')
                _person = self.split_by_transaction(data_index, _person)

            if len(_person):
                by_person.append(_person)

        self.by_person = by_person

    @staticmethod
    def insider_weekly(data):
        data['week'] = data.index.to_period('W')
        data = data.groupby(by='week').agg(
            Insider=('Insider', 'first'),
            Relationship=('Relationship', 'first'),
            Value=('Value', 'sum'),
            Share=('Share', 'sum'),
        )
        data['Cost'] = data.Value / data.Share
        data.index = data.index.to_timestamp()
        return data

    def get_by_person(self, weekly=False):
        if not self.by_person:
            self.prepare_insider_data()
        if not weekly:
            return self.by_person
        return [{k: InsiderDataFrame.insider_weekly(v) for k, v in _person.items()}
                for _person in self.by_person]
