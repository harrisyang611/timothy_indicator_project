import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import os
import talib as ta
import datetime

list_file = os.listdir('./data/')
list_file = [i for i in list_file if '_full' in i]
print(list_file)

import random


random.seed(611)

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

def tuning_model(X_train, y_train, pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=split, scoring=scoring, refit='sharpe', return_train_score=True)
    grid_search.fit(X_train, y_train.values.ravel())
    best_parameters = grid_search.best_params_
    best_model = grid_search.best_estimator_
    results = pd.DataFrame(grid_search.cv_results_)
    return(grid_search, results, grid_search.best_score_*100)

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


def model_evaluation(df, model_name, pipe, param_grid):
    df_model = df[base_col + ['retFut1']]
    X_train = df_model.drop(['retFut1'], axis=1)
    y_train = df_model[['retFut1']]

    grid_search, res, eva = tuning_model(X_train, y_train, pipe, param_grid)

    base_sharpe = np.max(res['mean_test_sharpe'])
    base_rmse = np.max(res['mean_test_rmse'])
    base_spearmanr = np.max(res['mean_test_spearmanr'])
    CAGR, Sharpe_ratio, Calmar = extra_model_eva(grid_search, X_train, y_train)

    ind_result = [{'col': 'base', 
            'model_name':model_name,
            'test_sharpe': base_sharpe ,
            'test_rmse': base_rmse, 
            'test_spearmanr': base_spearmanr,
            'train_cagr':CAGR,
            'train_sharpe_ratio':Sharpe_ratio,
            'calmar':Calmar
            }]

    for ind in indicator_col:
        df_temp = df[base_col + ['retFut1'] + [ind]]
        X_train_temp = df_temp.drop(['retFut1'], axis=1)
        y_train_temp = df_temp[['retFut1']]
        grid_search, cv_res, cv_score = tuning_model(X_train_temp, y_train_temp, pipe, param_grid)
        CAGR, Sharpe_ratio, Calmar = extra_model_eva(grid_search, X_train_temp, y_train_temp)
        ind_dict = {'col': ind, 
            'model_name':model_name,
            'test_sharpe': np.max(cv_res['mean_test_sharpe']) , 
            'test_rmse': np.max(cv_res['mean_test_rmse']), 
            'test_spearmanr': np.max(cv_res['mean_test_spearmanr']),
            'train_cagr':CAGR,
            'train_sharpe_ratio':Sharpe_ratio,
            'calmar':Calmar
            }
        ind_result.append(ind_dict)

    return(ind_result)





for my_ticker_file in list_file:
    
    ticker_name = my_ticker_file.split('_')[0]
    file_location = './data/' + my_ticker_file
    print(f'Start modelling for {ticker_name}: ',  datetime.datetime.now()
)

    df = pd.read_csv(file_location)
    df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d')

    df = df.set_index('Date')

    for n in list(range(1,30)):
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
        'ret10', 'ret11', 'ret12', 'ret13', 'ret14', 'ret15', 'ret16', 'ret17',
        'ret18', 'ret19', 'ret20', 'ret21', 'ret22', 'ret23', 'ret24', 'ret25',
        'ret26', 'ret27', 'ret28', 'ret29']
    # df_model = df[base_col + ['retFut1']]

    # X_train = df_model.drop(['retFut1'], axis=1)
    # y_train = df_model[['retFut1']]

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

    ridge_model_res = model_evaluation(df, 'ridge', ridge_pipe, ridge_param_grid)
    ridge_model_df = pd.DataFrame(ridge_model_res)

    print(f'ridge model finished: ',  datetime.datetime.now())

    from sklearn.ensemble import RandomForestRegressor
    rf_pipe = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('rf', RandomForestRegressor())])
    rf_param_grid = [{ 'rf__n_estimators': [100] , 'rf__max_depth':[10,15,20]}]

    rf_model_res = model_evaluation(df, 'rf', rf_pipe, rf_param_grid)
    rf_model_df = pd.DataFrame(rf_model_res)

    print(f'rf model finished: ',  datetime.datetime.now())

    from sklearn.ensemble import GradientBoostingRegressor
    gb_pipe = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('gb', GradientBoostingRegressor())])
    gb_param_grid = [{ 'gb__n_estimators': [100] , 'gb__max_depth':[5,10,15,20]}]

    gb_model_res = model_evaluation(df, 'gb', gb_pipe, gb_param_grid)
    gb_model_df = pd.DataFrame(gb_model_res)

    print(f'gb model finished: ',  datetime.datetime.now())

    myres = pd.concat([ridge_model_df, rf_model_df, gb_model_df])
    outfile = './data/' + ticker_name + '_model_result.csv'
    myres.to_csv(outfile, index=False)

    print(f'model trained, saved at {outfile} at: ',  datetime.datetime.now())
