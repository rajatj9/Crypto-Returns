import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_quantiles, parse_to_float, convert_date_format
import sys
from config import token_keyword_mapping

# filename = sys.argv[1]

filename = "btc.csv"

btc_data = pd.read_csv(filename, parse_dates=True)

btc_data = btc_data.reindex(index=btc_data.index[::-1])

btc_data['Date'] = btc_data['Date'].apply(lambda x: convert_date_format(x))


if 'btc' in filename or 'eth' in filename or 'bch' in filename or 'dash' in filename:
    btc_data['Open'] = btc_data['Open'].apply(parse_to_float)

# Daily Return Calculations
btc_data_daily = btc_data.copy()
btc_data_daily['Close'] = btc_data_daily['Open'].shift(-1)
btc_data_daily['ReturnD'] = btc_data_daily.apply(lambda x: (x['Close'] - x['Open'])/x['Open']*100, axis=1)
btc_data_daily = btc_data_daily.dropna()
btc_daily_graph_data = btc_data_daily[btc_data_daily['ReturnD'] < 100]

# Daily Return Plot
plt.hist(btc_daily_graph_data['ReturnD'], bins=200)
plt.title(f'Daily Return Distribution for {filename.split(".")[0].upper()}')
plt.xlabel('Daily Return x 100')
plt.ylabel('Frequency')
plt.show()

# Log Daily Return Plot
# plt.hist(np.log(btc_daily_graph_data['ReturnD']), bins=200)
# plt.title(f'Log Daily Return Distribution for {filename.split(".")[0].upper()}')
# plt.xlabel('Log Daily Return x 100')
# plt.ylabel('Frequency')
# plt.show()


# Weekly Return Calculations
btc_data_weekly = btc_data.copy()
btc_data_weekly = btc_data_weekly.iloc[::7, :]
btc_data_weekly["CloseT+7"] = btc_data_weekly['Open'].shift(-1)
btc_data_weekly['ReturnW'] = btc_data_weekly.apply(lambda x: (x['CloseT+7'] - x['Open'])/x['Open']*100, axis=1)
btc_data_weekly = btc_data_weekly.dropna()

# Weekly Return Plot
plt.hist(btc_data_weekly['ReturnW'], bins=100)
plt.title(f'Weekly Return Distribution for {filename.split(".")[0].upper()}')
plt.xlabel('Weekly Return x 100')
plt.ylabel('Frequency')
plt.show()

# Log Weekly Return Plot
# plt.hist(np.log(btc_data_weekly['ReturnW']), bins=200)
# plt.title(f'Log Weekly Return Distribution for {filename.split(".")[0].upper()}')
# plt.xlabel('Log Weekly Return x 100')
# plt.ylabel('Frequency')
# plt.show()


# Monthly return calculation
btc_data_monthly = btc_data.copy()
btc_data_monthly = btc_data_monthly.iloc[::30, :]
btc_data_monthly["CloseT+30"] = btc_data_monthly['Open'].shift(-1)
btc_data_monthly['ReturnM'] = btc_data_monthly.apply(lambda x: (x['CloseT+30'] - x['Open'])/x['Open']*100, axis=1)
btc_data_monthly = btc_data_monthly.dropna()

# Monthly Return Plot
plt.hist(btc_data_monthly['ReturnM'], bins=25)
plt.title(f'Monthly Return for Distribution {filename.split(".")[0].upper()}')
plt.xlabel('Monthly Return x 100')
plt.ylabel('Frequency')
plt.show()

# Log Monthly Return Plot
# plt.hist(np.log(btc_data_monthly['ReturnM']), bins=25)
# plt.title(f'Log Monthly Return for Distribution {filename.split(".")[0].upper()}')
# plt.xlabel('Log Monthly Return x 100')
# plt.ylabel('Frequency')
# plt.show()


# Plot of Returns
plt.plot(btc_data['Date'], btc_data['Open'])
plt.xticks(btc_data['Date'][::100],  rotation='vertical')
plt.title(f'Returns for {filename.split(".")[0].upper()}')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()


def add_google_data(tick_data, frequency):
    currency_name = token_keyword_mapping[filename.split('.')[0]]
    google_data = pd.read_csv(f"{filename.split('.')[0]}_trends.csv")
    google_data = google_data.iloc[::frequency, :]
    google_data = google_data[['date', currency_name]]
    google_data = google_data.rename(columns={token_keyword_mapping[filename.split('.')[0]]: 'Google', 'date': 'Date'})
    return tick_data.merge(google_data, how='left', on='Date')


btc_data_daily = add_google_data(btc_data_daily)

print(btc_data_daily.tail(10))

