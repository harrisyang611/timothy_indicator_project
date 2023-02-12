
import pandas as pd
import shutil
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime
import subprocess
import yfinance as yf
import time


yf.pdr_override()
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
    "^OEX","^NDX","^RUT","^DJI", ## index
    "TSLA","NKE","AMZN","WMT","JPM","GS","BA","CAT","IBM","MSFT","TLT", ## populars
    "BAC","KO","AAPL","DIS","SBUX" ## stable  https://www.fool.com/investing/stock-market/types-of-stocks/safe-stocks/
] 




for stock in stock_list:
    ## remove the ^ in front of the stock name, otherwise leave the same
    stock_name = stock[1:] if stock[0] == '^' else stock

    ## file location
    file_location = './data/' + stock_name + '.txt'
    ## get yahoo finance from start date to end date
    df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
    ## drop adjusted close
    df.drop(['Adj Close'], axis=1, inplace=True)
    print(stock)
    print(f'received data from yf for {stock} with {df.shape}')

    df = df.reset_index()
    ## change to date to datetime and format it
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y%m%d')
    ## write  it down to csv file because C++ single exe. only takes in flat files
    df.to_csv(file_location, header=None, index=None, sep=' ', mode='a')

    ## use popen to execute the single exe
    ## because the program will require us to key-input to kill the process
    ## we manually kill the program
    
    p=subprocess.Popen(['.\exe\single.exe', file_location, 'VS.txt'])
    time.sleep(4)
    p.kill()
    # # subprocess.call(['.\exe\single.exe', file_location, 'VS.txt'])

    print('enter is input, executable finishing')

    ## single will create file '.\OUTVARS.TXT' 
    ## we move to <ticker_name>_singal.txt file
    ## loaded the file merge with our original data

    signal_location = './data/' + stock_name + '_signal.txt'
    shutil.move('.\OUTVARS.TXT', signal_location)

    indicator = pd.read_csv(signal_location ,delim_whitespace=True)
    indicator['Date'] = pd.to_datetime(indicator['Date'], format = '%Y%m%d')
    indicator['Date'] = indicator['Date'].dt.strftime('%Y%m%d')

    print('indicator file has shape', indicator.shape)

    df_result = df.merge(indicator, left_on = 'Date', right_on = 'Date', how = 'left')
    df_result.to_csv('./data/' + stock_name + '_full.csv', index=False)

