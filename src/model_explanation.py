
import pandas as pd
import shap
import lime.lime_tabular
import numpy as np
import talib as ta
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import datetime


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


def shap_wrapper(model, X_train, plot_title, plot_location):
    explainer = shap.Explainer(model.predict, X_train)
    model_exp = explainer(X_train)
    plt.clf()
    shap.summary_plot(model_exp, features=X_train, feature_names=X_train.columns, show=False)
    plt.title(plot_title)
    plt.savefig(plot_location)
    plt.close()


def shap_plot(model, X_train, model_name,indicator_name, ticker_name):

    plot_title = f'{model_name}_{indicator_name}_indicator'
    plot_location = f'./output/{ticker_name}_{plot_title}_shap.png'

    shap_wrapper(model, X_train, plot_title, plot_location)



if __name__ == '__main__':

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
        {'col': 'base', 'model_name': 'gb', 'ticker_name': 'BA'}
        # {'col': 'PR_INT_20', 'model_name': 'gb', 'ticker_name': 'RUT'},
        # {'col': 'ENT_2_10', 'model_name': 'rf', 'ticker_name': 'RUT'},
        # {'col': 'MADIFF_10_100_0', 'model_name': 'ridge', 'ticker_name': 'RUT'},
        # {'col': 'STO_20_1', 'model_name': 'gb', 'ticker_name': 'RUT'},
        # {'col': 'PR_INT_20', 'model_name': 'rf', 'ticker_name': 'RUT'},
        # {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'RUT'},
        # {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'RUT'},
        # {'col': 'base', 'model_name': 'rf', 'ticker_name': 'RUT'},
        # {'col': 'base', 'model_name': 'gb', 'ticker_name': 'RUT'},
        # {'col': 'DT_RSI_2_20', 'model_name': 'gb', 'ticker_name': 'TSLA'},
        # {'col': 'LINTRND_10', 'model_name': 'rf', 'ticker_name': 'TSLA'},
        # {'col': 'RSI_20', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
        # {'col': 'ENT_4_16', 'model_name': 'gb', 'ticker_name': 'TSLA'},
        # {'col': 'PR_INT_0', 'model_name': 'rf', 'ticker_name': 'TSLA'},
        # {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
        # {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'TSLA'},
        # {'col': 'base', 'model_name': 'rf', 'ticker_name': 'TSLA'},
        # {'col': 'base', 'model_name': 'gb', 'ticker_name': 'TSLA'},
        # {'col': 'DT_RSI_2_20', 'model_name': 'gb', 'ticker_name': 'BAC'},
        # {'col': 'LINTRND_10', 'model_name': 'rf', 'ticker_name': 'BAC'},
        # {'col': 'RSI_20', 'model_name': 'ridge', 'ticker_name': 'BAC'},
        # {'col': 'ENT_4_16', 'model_name': 'gb', 'ticker_name': 'BAC'},
        # {'col': 'PR_INT_0', 'model_name': 'rf', 'ticker_name': 'BAC'},
        # {'col': 'PR_INT_0', 'model_name': 'ridge', 'ticker_name': 'BAC'},
        # {'col': 'base', 'model_name': 'ridge', 'ticker_name': 'BAC'},
        # {'col': 'base', 'model_name': 'rf', 'ticker_name': 'BAC'},
        # {'col': 'base', 'model_name': 'gb', 'ticker_name': 'BAC'}
        ]

    for eva_dict in evaluation_list:

        indicator = eva_dict['col']
        model_name = eva_dict['model_name']
        ticker_name = eva_dict['ticker_name']

        print(f'Model explanation Shap plots started for {ticker_name} with {model_name} and {indicator}: ',  datetime.datetime.now())

        df = pd.read_csv(f'./data/{ticker_name}_full.csv')
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d')

        df = df.set_index('Date')

        for n in list(range(1,30)):
            name = 'ret' + str(n)
            df[name] = df['Open'].pct_change(periods=n)#for trading with open

        df['retFut1'] = df['Open'].pct_change(1).shift(-1).fillna(0)
        df['RSI_20_ta'] = ta.RSI(np.array(df['Open']), timeperiod = 20)
        df['RSI_25_ta'] = ta.RSI(np.array(df['Open']), timeperiod = 25)


        base_col = ['ret1', 'ret2', 'ret3', 'ret4', 'ret5', 'ret6', 'ret7', 'ret8', 'ret9',
            'ret10', 'ret11', 'ret12', 'ret13', 'ret14', 'ret15', 'ret16', 'ret17',
            'ret18', 'ret19', 'ret20', 'ret21', 'ret22', 'ret23', 'ret24', 'ret25',
            'ret26', 'ret27', 'ret28', 'ret29']
        
        split = TimeSeriesSplit(n_splits=5)
        sharpe_scorer = make_scorer(sharpe, greater_is_better=True)
        spearmanr_scorer = make_scorer(information_coefficient, greater_is_better=True)
        scoring = {"rmse": "neg_root_mean_squared_error", 'sharpe': sharpe_scorer, 'spearmanr': spearmanr_scorer}
        numeric_sub_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value = 0)),
            ('scaler', StandardScaler())])

        ## training here
        if(model_name == 'ridge'):
            ridge = Ridge(max_iter=5000) 
            a_rs = np.logspace(-40, 0, num=100, endpoint = True)

            pipeline = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('ridge', ridge)])
            param_grid = [{ 'ridge__alpha': a_rs }]
        
        elif(model_name == 'rf'):
            
            pipeline = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('rf', RandomForestRegressor())])
            param_grid = [{ 'rf__n_estimators': [100] , 'rf__max_depth':[10,15,20]}]

        else:
            pipeline = Pipeline(steps=[('preprocessor', numeric_sub_pipeline),('gb', GradientBoostingRegressor())])
            param_grid = [{ 'gb__n_estimators': [100] , 'gb__max_depth':[5,10,15,20]}]

        if(indicator == 'base'):
            df_model = df[base_col + ['retFut1']]
            X_train = df_model.drop(['retFut1'], axis=1)
            y_train = df_model[['retFut1']]
        
        else:
            df_model = df[base_col + [indicator,'retFut1']]
            X_train = df_model.drop(['retFut1'], axis=1)
            y_train = df_model[['retFut1']]

        grid_search = GridSearchCV(pipeline, param_grid, cv=split, scoring=scoring, refit='sharpe', return_train_score=True)
        grid_search.fit(X_train, y_train.values.ravel())

        print(f'Model trained finish: ',  datetime.datetime.now())

        shap_plot(grid_search, X_train, model_name, indicator, ticker_name)

        print(f'Model explanation Shap plots finished for {ticker_name} with {model_name} and {indicator}: ',  datetime.datetime.now())


        





    