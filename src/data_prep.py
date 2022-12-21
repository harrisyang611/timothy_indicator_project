
import pandas as pd
import shutil
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime
import subprocess
import yfinance as yf
import time



yf.pdr_override() # <== that's all it takes :-)
start_date=datetime.datetime(2010, 1, 9)
end_date=datetime.datetime(2022, 1, 9)

## OEX: S&P
## NDX: nasdaq
## RUT: russell 2000
## NKE: Nike
## JPM : JP morgan
## BAC: Bank of america
## KO: coke-cola

## DJI list : #stock_list = ['AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','RTX','UNH', 'VZ','V','WMT','WBA', 'TLT']
## industrial list : #stock_list = ["FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ","TLT"]
## 10 sectors list : #stock_list = ["XLB","XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XTR", "TLT"] 



stock_list = [
    # "DB"
    "^OEX","^NDX","^RUT","^DJI", ## index
    "TSLA","NKE","AMZN","WMT","JPM","GS","BA","CAT","IBM","MSFT","TLT", ## populars
    "BAC","KO","AAPL","DIS","SBUX" ## stable  https://www.fool.com/investing/stock-market/types-of-stocks/safe-stocks/
]  


# stock_str = ""
# for i in stock_list:
#     stock_str  = stock_str +  i + "."
# print(stock_str)


for stock in stock_list:
    stock_name = stock[1:] if stock[0] == '^' else stock
    file_location = './data/' + stock_name + '.txt'
    df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
    df.drop(['Adj Close'], axis=1, inplace=True)
    print(stock)
    print(f'received data from yf for {stock} with {df.shape}')
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y%m%d')
    df.to_csv(file_location, header=None, index=None, sep=' ', mode='a')

    ## use popen to kill the program
    p=subprocess.Popen(['.\exe\single.exe', file_location, 'VS.txt'])
    time.sleep(4)
    p.kill()
    # # subprocess.call(['.\exe\single.exe', file_location, 'VS.txt'])

    print('enter is input, executable finishing')

    signal_location = './data/' + stock_name + '_signal.txt'
    shutil.move('.\OUTVARS.TXT', signal_location)

    indicator = pd.read_csv('.\data\OEX_signal.txt' ,delim_whitespace=True)
    indicator['Date'] = pd.to_datetime(indicator['Date'], format = '%Y%m%d')
    indicator['Date'] = indicator['Date'].dt.strftime('%Y%m%d')

    print('indicator file has shape', indicator.shape)

    df_result = df.merge(indicator, left_on = 'Date', right_on = 'Date', how = 'left')
    df_result.to_csv('./data/' + stock_name + '_full.csv', index=False)


