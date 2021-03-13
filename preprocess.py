
import os
import numpy as np
import pandas as pd
import warnings
import config

warnings.filterwarnings("ignore", category=DeprecationWarning)


def preprocess(data):
    for i, col1 in enumerate(config.dates):
        for col2 in config.dates[i+1:]:
            col1_is_bigger_df = data.loc[data[col1] > data[col2]]
            if len(col1_is_bigger_df) > 0:
                data.drop(labels=col1_is_bigger_df.index, inplace=True)

    data.insert(loc=0, value=(data.MaturityDate - data.DealDate).astype('timedelta64[D]'), column='DealDurationDays')
    data.insert(loc=1, value=(data.DealDurationDays // 30).astype(int), column='DealDurationMonths')
    data = data.sort_values(by='DealDate').reset_index(drop=True)
    return data

if __name__ == '__main__':
    pass