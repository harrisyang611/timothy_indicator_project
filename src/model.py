import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import shap
import matplotlib.pyplot as plt


import os
import talib as ta
import datetime
from tqdm import tqdm
import random
from joblib import dump, load


random.seed(611)

list_file = os.listdir('./data/')
list_file = [i for i in list_file if '_full' in i]
print(list_file)
output_dir = './output_new/' 

evaluation_list = [
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'TLT'},
    {'col': 'RSI_20_ta', 'model_name': 'rf', 'ticker_name': 'TLT'},
    {'col': 'FTI_BF', 'model_name': 'ridge', 'ticker_name': 'TLT'},
    {'col': 'RSI_20_ta', 'model_name': 'gb', 'ticker_name': 'TLT'},
    {'col': 'PR_INT_20', 'model_name': 'rf', 'ticker_name': 'TLT'},
    {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'TLT'},
    {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'TLT'},
    {'col': 'base', 'model_name': 'rf', 'ticker_name': 'TLT'},
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'TLT'},
    {'col': 'DT_RSI_2_20', 'model_name': 'gb', 'ticker_name': 'BA'},
    {'col': 'ENT_4_16', 'model_name': 'rf', 'ticker_name': 'BA'},
    {'col': 'RSI_25', 'model_name': 'ridge', 'ticker_name': 'BA'},
    {'col': 'MADIFF_10_100', 'model_name': 'gb', 'ticker_name': 'BA'},
    {'col': 'CMMA_10_252', 'model_name': 'rf', 'ticker_name': 'BA'},
    {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'BA'},
    {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'BA'},
    {'col': 'base', 'model_name': 'rf', 'ticker_name': 'BA'},
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'BA'},
    {'col': 'PR_INT_20', 'model_name': 'gb', 'ticker_name': 'RUT'},
    {'col': 'ENT_2_10', 'model_name': 'rf', 'ticker_name': 'RUT'},
    {'col': 'MADIFF_10_100_0', 'model_name': 'ridge', 'ticker_name': 'RUT'},
    {'col': 'STO_20_1', 'model_name': 'gb', 'ticker_name': 'RUT'},
    {'col': 'PR_INT_20', 'model_name': 'rf', 'ticker_name': 'RUT'},
    {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'RUT'},
    {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'RUT'},
    {'col': 'base', 'model_name': 'rf', 'ticker_name': 'RUT'},
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'RUT'},
    {'col': 'DT_RSI_2_20', 'model_name': 'gb', 'ticker_name': 'TSLA'},
    {'col': 'LINTRND_10', 'model_name': 'rf', 'ticker_name': 'TSLA'},
    {'col': 'RSI_20', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
    {'col': 'ENT_4_16', 'model_name': 'gb', 'ticker_name': 'TSLA'},
    {'col': 'PR_INT_0', 'model_name': 'rf', 'ticker_name': 'TSLA'},
    {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
    {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
    {'col': 'base', 'model_name': 'rf', 'ticker_name': 'TSLA'},
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'TSLA'},
    {'col': 'DT_RSI_2_20', 'model_name': 'gb', 'ticker_name': 'BAC'},
    {'col': 'LINTRND_10', 'model_name': 'rf', 'ticker_name': 'BAC'},
    {'col': 'RSI_20', 'model_name': 'ridge', 'ticker_name': 'BAC'},
    {'col': 'ENT_4_16', 'model_name': 'gb', 'ticker_name': 'BAC'},
    {'col': 'PR_INT_0', 'model_name': 'rf', 'ticker_name': 'BAC'},
    {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'BAC'},
    {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'BAC'},
    {'col': 'base', 'model_name': 'rf', 'ticker_name': 'BAC'},
    {'col': 'base', 'model_name': 'gb', 'ticker_name': 'BAC'}
    ]


def information_coefficient(y_true, y_pred):
    rho, pval = spearmanr(y_true,y_pred) #spearman's rank correlation
    # print (rho)
    return rho

def sharpe(y_true, y_pred):
    positions = np.where(y_pred> 0,1,-1 )
    dailyRet = pd.Series(positions).shift(1).fillna(0).values * y_true
    dailyRet = np.nan_to_num(dailyRet)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    return ratio


def calculateMaxDD(cumret):
    highwatermark = np.zeros(len(cumret))
    drawdown      = np.zeros(len(cumret))
    drawdownduration = np.zeros(len(cumret))
    for t in range(1, len(cumret)):
        highwatermark[t] = np.max([highwatermark[t-1], cumret[t]])
        drawdown[t] = (1+cumret[t]) / (1 + highwatermark[t]) - 1
        if (drawdown[t]==0):
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1
    return np.min(drawdown), np.max(drawdownduration)


def extra_model_eva(grid_search, X, y):
    positions = np.where(grid_search.predict(X)> 0,1,-1 ) #POSITIONS
    dailyRet = pd.Series(positions).fillna(0).values * y.retFut1 #for trading right after the open
    dailyRet = dailyRet.fillna(0)
    cumret = np.cumprod(dailyRet + 1) - 1
    cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
    maxDD, maxDDD = calculateMaxDD(cumret)
    ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
    ## return CAGR, Sharpe ratio, Calmar
    return(cagr, ratio, -cagr/maxDD)


def tuning_model(X_train, y_train, pipeline, param_grid, col):
    grid_search = GridSearchCV(pipeline, param_grid, cv=split, scoring=scoring, refit='sharpe', return_train_score=True)
    grid_search.fit(X_train, y_train.values.ravel())
    best_parameters = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    results = pd.DataFrame(grid_search.cv_results_)

    base_sharpe = np.max(results['mean_test_sharpe'])
    base_rmse = np.max(results['mean_test_rmse'])
    base_spearmanr = np.max(results['mean_test_spearmanr'])
    CAGR, Sharpe_ratio, Calmar = extra_model_eva(grid_search, X_train, y_train)

    res_dict = {
        'col': col, 
        'tuning_sharpe': base_sharpe ,
        'tuning_rmse': base_rmse, 
        'tuning_spearmanr': base_spearmanr,
        'train_cagr':CAGR,
        'train_sharpe_ratio':Sharpe_ratio,
        'train_calmar':Calmar,
        'model_tuned':grid_search,
        'column_names':X_train.columns
    }
    return(res_dict, grid_search)


def model_building(df, model_name, pipe, param_grid, train_size = 0.75):

    no_rows = df.shape[0]
    train_size = int(no_rows * train_size)

    df_model = df[base_col + ['retFut1']]
    train, test = df_model[0:train_size], df_model[train_size: no_rows]
    X_train = train.drop(['retFut1'], axis=1)
    y_train = train[['retFut1']]
    X_test = test.drop(['retFut1'], axis=1)
    y_test = test['retFut1']
    

    indicator_result = []
    model_result, tuned_model = tuning_model(X_train, y_train, pipe, param_grid, 'base')

    y_pred = tuned_model.predict(X_test)
    test_spearmanr = information_coefficient(y_test, y_pred)
    test_sharpe = sharpe(y_test, y_pred)
    test_rmse = mean_squared_error(y_test, y_pred)
    model_result['test_spearmanr'] = test_spearmanr
    model_result['test_sharpe'] = test_sharpe
    model_result['test_rmse'] = test_rmse

    indicator_result.append(model_result)

    for ind in indicator_col:
        df_temp = df[base_col + ['retFut1'] + [ind]]
        train, test = df_temp[0:train_size], df_temp[train_size: no_rows]
        X_train = train.drop(['retFut1'], axis=1)
        y_train = train[['retFut1']]
        X_test = test.drop(['retFut1'], axis=1)
        y_test = test[['retFut1']]

        model_result, tuned_model = tuning_model(X_train, y_train, pipe, param_grid, ind)
        y_pred = tuned_model.predict(X_test)
        test_spearmanr = information_coefficient(y_test, y_pred)
        test_sharpe = sharpe(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred)
        model_result['test_spearmanr'] = test_spearmanr
        model_result['test_sharpe'] = test_sharpe
        model_result['test_rmse'] = test_rmse
        indicator_result.append(model_result)

    model_df = pd.DataFrame(indicator_result)
    model_df['model_name'] = model_name

    return( model_df )


def shap_wrapper(model, X_train, plot_title, plot_location):
    explainer = shap.Explainer(model.predict, X_train)
    model_exp = explainer(X_train)
    plt.clf()
    shap.summary_plot(model_exp, features=X_train, feature_names=X_train.columns, show=False)
    plt.title(plot_title)
    plt.savefig(plot_location)
    plt.close()


def shap_plot(model, df, model_name,indicator_name, ticker_name, outdir = output_dir):
    
    no_rows = df.shape[0]
    train_size = int(no_rows * 0.75)
    X_train, test = df[0:train_size], df[train_size: no_rows]

    plot_title = f'{model_name}_{indicator_name}_indicator'
    plot_location = f'{outdir}/{ticker_name}_{plot_title}_shap.png'

    shap_wrapper(model, X_train, plot_title, plot_location)


for my_ticker_file in tqdm(list_file):
    
    ticker_name = my_ticker_file.split('_')[0]
    file_location = './data/' + my_ticker_file
    print(f'Start modelling for {ticker_name}: ',  datetime.datetime.now())

    df = pd.read_csv(file_location)
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d')

    df = df.set_index('Date')

    for n in list(range(1,15)):
        name = 'ret' + str(n)
        df[name] = df['Open'].pct_change(periods=n)#for trading with open

    df['retFut1'] = df['Open'].pct_change(1).shift(-1).fillna(0)
    
    df['RSI_20_ta'] = ta.RSI(np.array(df['Open']), timeperiod = 20)
    df['RSI_25_ta'] = ta.RSI(np.array(df['Open']), timeperiod = 25)

    indicator_col = [
        'RSI_20','RSI_20_ta', 'RSI_25','RSI_25_ta',
        'DT_RSI_2_20', 'STO_20_1', 'MADIFF_10_100_0', 'MADIFF_10_100',
        'MACD_10_100_5', 'LINTRND_10', 'PR_INT_0', 'PR_INT_20', 'CMMA_10_252',
        'ENT_2_10', 'ENT_4_16', 'FTI_LP', 'FTI_BP', 'FTI_BF'
    ]
    base_col = ['ret1', 'ret2', 'ret3', 'ret4', 'ret5', 'ret6', 'ret7', 'ret8', 'ret9',
        'ret10', 'ret11', 'ret12', 'ret13', 'ret14']
        # , 'ret15', 'ret16', 'ret17',
        # 'ret18', 'ret19', 'ret20', 'ret21', 'ret22', 'ret23', 'ret24', 'ret25',
        # 'ret26', 'ret27', 'ret28', 'ret29']

    print(f'Data Prep finished: ',  datetime.datetime.now())

    ## model def

    sharpe_scorer = make_scorer(sharpe, greater_is_better=True)
    spearmanr_scorer = make_scorer(information_coefficient, greater_is_better=True)
    scoring = {"rmse": "neg_root_mean_squared_error", 'sharpe': sharpe_scorer, 'spearmanr': spearmanr_scorer}
    split = TimeSeriesSplit(n_splits=5)

    ## training here
    numeric_sub_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value = 0)),
    ('scaler', StandardScaler())])
    ridge = Ridge(max_iter=5000) 
    a_rs = np.logspace(-40, 0, num=100, endpoint = True)

    ridge_pipe = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('ridge', ridge)])
    ridge_param_grid = [{ 'ridge__alpha': a_rs }]

    ridge_df = model_building(df, 'ridge', ridge_pipe, ridge_param_grid)
    
    print(f'ridge model finished: ',  datetime.datetime.now())

    rf_pipe = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('rf', RandomForestRegressor())])
    rf_param_grid = [{ 'rf__n_estimators': [250] , 'rf__max_depth':[10,20,30]}]

    rf_df = model_building(df, 'rf', rf_pipe, rf_param_grid)

    print(f'rf model finished: ',  datetime.datetime.now())

    gb_pipe = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('gb', GradientBoostingRegressor())])
    gb_param_grid = [{ 'gb__n_estimators': [250] , 'gb__max_depth':[10,20,30]}]

    db_df = model_building(df, 'gb', gb_pipe, gb_param_grid)

    print(f'gb model finished: ',  datetime.datetime.now())

    myres = pd.concat([ridge_df , rf_df, db_df])
    for index, row in myres.iterrows():
        model_tuned = row['model_tuned']
        out_model_name = output_dir + ticker_name +'_'+ row['model_name'] + '_' + row['col'] + '.joblib'
        if not os.path.exists(out_model_name):
            dump(model_tuned, out_model_name) 
        
        try:
            for interest_combo in evaluation_list:
                if( row['model_name'] == interest_combo['model_name'] and  
                    row['col'] == interest_combo['col'] and
                    ticker_name == interest_combo['ticker_name'] ):

                    shap_plot(model_tuned, df[row['column_names']], row['model_name'], row['col'] , ticker_name)
                    print(f'shap created for {interest_combo} at: ',  datetime.datetime.now())
        except:
            pass


    myres = myres.drop(['model_tuned','column_names'], axis=1)
    outfile = output_dir + ticker_name + '_model_result.csv'
    myres.to_csv(outfile, index=False)

    print(f'model trained, saved at {outfile} at: ',  datetime.datetime.now())
