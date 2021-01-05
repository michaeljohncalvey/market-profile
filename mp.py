import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import datetime

from data_loader import Data


def create_split_market_profile(ticker, frequency='H', start_date=None, end_date=None, height_precision=1):
    # Default to 1 week if no input given:
    if start_date is None:
        start_date = datetime.date.today() - datetime.timedelta(weeks=1)
        # start_date = start_date.replace(day=1)
    if end_date is None:
        end_date = datetime.date.today()

    raw_data = Data.load_data_from_file(ticker, '30min', start_date=start_date, end_date=end_date)
    raw_data[('high')] = raw_data[("high")] * height_precision
    raw_data[('low')] = raw_data[("low")] * height_precision

    time_groups = raw_data.groupby(pd.Grouper(freq=frequency))['close'].mean()
    current_time_group_index = 0
    mp = defaultdict(str)
    char_mark = 64  # Starts us off on 'A'

    # Build dict with all needed prices
    tot_min_price = min(np.array(raw_data['low']))
    tot_max_price = max(np.array(raw_data['high']))
    for price in range(int(tot_min_price), int(tot_max_price)):
        mp[price]+=('\t')

    # add max price as for loop above ignores it
    mp[tot_max_price] = '\t' + str(time_groups.index[current_time_group_index])[5:7] + '/' + str(time_groups.index[current_time_group_index])[3:4]

    for x in range(0, len(raw_data)-1):
        # print(current_time_group_index)
        # print(raw_data.index)
        if raw_data.index[x] > time_groups.index[current_time_group_index]:
            # new time period
            char_mark = 64
            # buffer and tab all entries
            buffer_max = max([len(v) for k, v in mp.items()])
            # if current_time_group_index < len(time_groups)-1:
            current_time_group_index += 1
            for k, v in mp.items():
                mp[k] += (chr(32) * (buffer_max - len(mp[k]))) + '\t'
            print(time_groups.index[current_time_group_index])
            mp[tot_max_price] += str(time_groups.index[current_time_group_index])[5:7] + '/' + str(time_groups.index[current_time_group_index])[3:4]

        # print(str(char_mark) + chr(char_mark))
        char_mark += 1
        min_price = raw_data['low'][x]
        max_price = raw_data['high'][x]
        for price in range(int(min_price), int(max_price)):
            mp[price] += (chr(char_mark))

    sorted_keys = sorted(mp.keys(), reverse=True)
    for x in sorted_keys:
        # buffer each list
        # print(str(x) + ": \t" + ''.join(mp[x]))
        print(str("{0:.2f}".format((x * 1.0) / height_precision)) + ': \t' + ''.join(mp[x]))


def print_grouped_market_profile(ticker, height_precision=1, frequency="30min", start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.date.today() - datetime.timedelta(days=4)
    if end_date is None:
        end_date = datetime.date.today()
    raw_data = Data.load_data_from_file(ticker, '30min', start_date=start_date, end_date=end_date)

    mp = defaultdict(str)
    char_mark = 64

    for index, row in raw_data.iterrows():
        char_mark += 1
        if char_mark == 91:
            char_mark = 97
        elif char_mark == 123:
            char_mark = 65
        min_price = np.round(row['low']) * height_precision
        max_price = np.round(row['high']) * height_precision

        for price in range(int(min_price), int(max_price+1)):
            mp[price] += (chr(char_mark))

    sorted_keys = sorted(mp.keys(), reverse=True)

    for x in sorted_keys:
        print(str("{0:.2f}".format((x * 1.0) / height_precision)) + ': \t' + ''.join(mp[x]))


def main():
    if len(sys.argv[1:]) == 1:
        symbol = sys.argv[1:][0]
        print_grouped_market_profile(symbol)
    elif len(sys.argv[1:]) == 2:
        symbol = sys.argv[1:][0]
        height_precision = float(sys.argv[1:][1])
        print_grouped_market_profile(symbol, height_precision)
    elif len(sys.argv[1:]) == 3:
        symbol = sys.argv[1:][0]
        height_precision = float(sys.argv[1:][1])
        frequency = sys.argv[1:][2]
        print_grouped_market_profile(symbol, height_precision, frequency)
        

if __name__ == "__main__":
    main()
