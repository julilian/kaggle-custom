import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import gc
import json
from pandas.io.json import json_normalize
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids



def load_df(csv_path='/Users/julianne/Desktop/kaggle-customer/input/train.csv', nrows=100):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
if __name__ == '__main__' :

    train_path = '/Users/julianne/Desktop/kaggle-customer/input/train.csv'
    test_path = '/Users/julianne/Desktop/kaggle-customer/input/test.csv'
    train_df = load_df(train_path)
    test_df = load_df(test_path)

    y_reg = train_df['totals.transactionRevenue'].fillna(0)

    train = train_df[['channelGrouping', 'date', 'fullVisitorId', 'sessionId','visitId', 'visitNumber', 'visitStartTime',
           'device.browser', 'device.deviceCategory' , 'device.isMobile','device.operatingSystemVersion',
           'geoNetwork.city','geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro','geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
            'totals.newVisits','totals.bounces',     'totals.hits', 'totals.pageviews', 'totals.visits',
           'trafficSource.isTrueDirect','trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source']]

    for df in [train, test]:
        df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
        df['sess_date_dow'] = df['vis_date'].dt.dayofweek
        df['sess_date_hours'] = df['vis_date'].dt.hour
        df['sess_date_dom'] = df['vis_date'].dt.day
        df.sort_values(['fullVisitorId', 'vis_date'], ascending=True, inplace=True)
        df['next_session_1'] = (
                                   df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')[
                                       'vis_date'].shift(1)
                               ).astype(np.int64) // 1e9 // 60 // 60
        df['next_session_2'] = (
                                   df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')[
                                       'vis_date'].shift(-1)
                               ).astype(np.int64) // 1e9 // 60 // 60


        df['nb_pageviews'] = df['date'].map(
            df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
        )

        df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']

    excluded_features = [ 'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits']
    categorical_features = [ _f for _f in train.columns    if (_f not in excluded_features) & (train[_f].dtype == 'object')]

    for f in categorical_features:
        train[f], indexer = pd.factorize(train[f])
        test[f] = indexer.get_indexer(test[f])

    folds = get_folds(df=train, n_splits=5)
    train_features = [_f for _f in train.columns if _f not in excluded_features]
    print(train_features)

    importances = pd.DataFrame()
    oof_reg_preds = np.zeros(train.shape[0])
    sub_reg_preds = np.zeros(test.shape[0])
    for fold_, (trn_, val_) in enumerate(folds):
        trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
        val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]

        reg = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.9,
            colsample_bytree=.9,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(val_x, np.log1p(val_y))],
            early_stopping_rounds=50,
            verbose=100,
            eval_metric='rmse'
        )
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_reg_preds[oof_reg_preds < 0] = 0
        _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_reg_preds += np.expm1(_preds) / len(folds)

    mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5

    import warnings
    warnings.simplefilter('ignore', FutureWarning)

    importances['gain_log'] = np.log1p(importances['gain'])
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

