import numpy as np
import pandas as pd
import datetime


def convert_date_format(date):
    date_time_obj = datetime.datetime.strptime(date, '%b %d, %Y')
    return date_time_obj.strftime('%Y-%m-%d')


def parse_to_float(string):
    string = string.replace(',', '')
    return float(string)


def apply_quantiles(x, bins=10):
    # calculate quantiles (breakpoints)
    x = pd.Series(x)
    quantiles = np.quantile(
        x[x.notnull()],
        np.linspace(0, 1, bins + 1)
    )
    quantiles[0] = x.min() - 1
    quantiles[-1] = x.max() + 1

    # cut the data a bit more
    return quantiles, pd.cut(x, quantiles, labels=False, duplicates='drop')
