import pandas as pd
import numpy as np
from collections import Counter,defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import pickle
from gensim.models import Word2Vec
from tqdm import tqdm
import sys
import os
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

train = pd.read_csv('../xfdata/创意视角下的数字广告CTR预估挑战赛复赛数据/train_data.txt', encoding='gbk', header=None,
                   )
train.columns = ['label', 'slotid', 'pkgname', 'ver', 'mediaid', 'material'] + [f'emb_{i}' for i in range(240)]

train2 = pd.read_csv('../xf/data/train_data.txt', encoding='gbk', header=None,
                    skiprows=[1613993, 1895025]
                   )
train2.columns = ['label', 'slotid', 'pkgname', 'ver', 'mediaid', 'material'] + [f'emb_{i}' for i in range(240)]
train = pd.concat([train2, train]).reset_index(drop=True)

test = pd.read_csv('../xfdata/创意视角下的数字广告CTR预估复赛测试集数据.txt', encoding='gbk', header=None)
test.columns = ['slotid', 'pkgname', 'ver', 'mediaid', 'material'] + [f'emb_{i}' for i in range(240)]

data = pd.concat([train, test]).reset_index(drop=True)

cols = [i for i in data.columns if i not in ['label',  ]]

tmp = data[data.duplicated(keep=False)]
tmp = tmp.reset_index()
data = data.reset_index()

tmp['count'] = tmp.groupby(cols)['index'].transform('count')
data = pd.merge(data, tmp[['index', 'count']], on='index', how='left')
data['count'].fillna(1, inplace=True)

unique_list = []
for f in [f'emb_{i}' for i in range(240)]:
    if data[f].nunique() < 20:
        unique_list.append(f)

for f in tqdm(unique_list):
    del data[f]

sparse_features = ['slotid', 'pkgname', 'ver', 'mediaid', ]

cat_cols = ['slotid', 'pkgname', 'ver', 'mediaid', 'material']

for f in cat_cols:
    data[f] = data[f].astype(str)
    data[f + '_count'] = data[f].map(data[f].value_counts())

for f in cat_cols[1:]:
    data[f'slotid_{f}'] = data['slotid'] + data[f]
    sparse_features.append(f'slotid_{f}')
    data[f'slotid_{f}_count'] = data[f'slotid_{f}'].map(data[f'slotid_{f}'].value_counts())

for feat in tqdm(sparse_features):
    lb = LabelEncoder()
    data[feat] = lb.fit_transform(data[feat])

data['material'] = data['material'].astype('int')

def group_fea(df, key, target):
    # 计算每一个key和多少个不同的target交互
    tmp = df.groupby(key, as_index=False)[target].agg({
        key + '_' + target + '_nunique': 'nunique',
    }).reset_index().drop('index', axis=1)

    return tmp
nunique_group = []
key_feature = ['slotid']
target_feature = ['pkgname', 'ver', 'mediaid',
                   'material']

for feat_1 in key_feature:
    for feat_2 in tqdm(target_feature):
        if feat_1 + '_' + feat_2 + '_nunique' not in nunique_group:
            nunique_group.append(feat_1 + '_' + feat_2 + '_nunique')
            tmp = group_fea(data, feat_1, feat_2)
            data = data.merge(tmp, on=feat_1, how='left')

        if feat_2 + '_' + feat_1 + '_nunique' not in nunique_group:
            nunique_group.append(feat_2 + '_' + feat_1 + '_nunique')
            tmp = group_fea(data, feat_2, feat_1)
            data = data.merge(tmp, on=feat_2, how='left')


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in tqdm(features):

        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


features = [i for i in data.columns if i not in ['label',
                                                 ]]

data = reduce_mem_usage(data)

for f1 in cat_cols:
    for f in tqdm([f'emb_{i}' for i in range(120)]):
        if f not in unique_list:
            data[f'{f}_{f1}_mean'] = data.groupby([f1])[f].transform('mean').astype('float16')
            data[f'{f}_{f1}_std'] = data.groupby([f1])[f].transform('std').astype('float16')

train = data[~data['label'].isna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)
del data
features = [i for i in train.columns if i not in ['label',  ]]
y = train['label']
print(len(features))

import catboost
import xgboost as xgb
import gc

lgb_params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': 2022,
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': -1,
    'lambda_l2': 2,
    'min_data_in_leaf': 40,
    'early_stopping_rounds': 50,
}

cat_params = {
    'eval_metric': 'AUC',
    'random_seed': 666,
    'logging_level': 'Verbose',
    'use_best_model': True,
    'loss_function': 'Logloss',
    'task_type': 'GPU',
    'learning_rate': 0.1
}


def train_model(X_train, X_test, features, y, model='lgb', folds=5, seed=2021, save_model=False):
    """
    Args:
        X_train: Training data (pd.DataFrame)
        X_test: Test data (pd.DataFrame)
        features: Features which is used to train the model (list)
        y: The labels of training data (list | pd.Series)
        folds: Folds for spliting the training data (int)
        seed: Random seed (int)
        save_model: Flag of saving the model (bool)

    Returns:
        feat_imp_df: Feature importance (pd.DataFrame)
        oof: Out of folds predictions (ndarray)
        predictions: Test data predictions (ndarray)

    """
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    oof = np.zeros(len(X_train))
    predictions = np.zeros((len(X_test)))

    if model == 'lgb':
        for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
            print("####" * 10)
            print("fold n°{}".format(fold_))
            trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
            val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
            clf = lgb.train(
                lgb_params,
                trn_data,
                valid_sets=[trn_data, val_data],
                verbose_eval=100,
                early_stopping_rounds=50,
                num_boost_round=10500,
                feval=lgb_amex_metric,
                # categorical_feature=cat_cols
            )

            oof[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
            predictions[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / folds
            feat_imp_df['imp'] += clf.feature_importance() / folds
            if save_model:
                clf.save_model(f'model_{fold_}.txt')


    elif model == 'xgb':

        dtest = xgb.DMatrix(X_test[features])
        for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
            print("####" * 10)
            print("fold n°{}".format(fold_))
            trn_data, trn_label = X_train.iloc[trn_idx][features], y.iloc[trn_idx]
            val_data, val_label = X_train.iloc[val_idx][features], y.iloc[val_idx]

            dtrain = xgb.DMatrix(data=X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
            dvalid = xgb.DMatrix(data=X_train.iloc[val_idx][features], label=y.iloc[val_idx])

            params = {
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist',
                'max_depth': 7,
                'subsample': 0.88,
                'colsample_bytree': 0.5,
                'gamma': 1.5,
                'min_child_weight': 8,
                'lambda': 70,
                'eta': 0.03,
                # 'booster': 'dart'
            }
            # watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            watchlist = [(dvalid, 'eval')]
            clf = xgb.train(params, dtrain=dtrain,
                            num_boost_round=5000, evals=watchlist,
                            early_stopping_rounds=50, maximize=False,
                            verbose_eval=100)
            print('best ntree_limit:', clf.best_ntree_limit)
            print('best score:', clf.best_score)

            oof[val_idx] = clf.predict(dvalid, iteration_range=(0, clf.best_ntree_limit))
            predictions[:] += clf.predict(dtest, iteration_range=(0, clf.best_ntree_limit))

            # feat_imp_df['imp'] += clf.get_score().values / folds
            if save_model:
                clf.save_model(f'model_{fold_}.json')

            del dtrain, dvalid, trn_data, val_data
            del clf, watchlist, trn_label, val_label
            gc.collect()
            print("fold n°{} finished. ".format(fold_))

    else:

        for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
            print("####" * 10)
            print("fold n°{}".format(fold_))
            trn_data = catboost.Pool(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
            val_data = catboost.Pool(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
            num_round = 10000
            clf = catboost.train(
                params=cat_params,
                pool=trn_data,
                iterations=num_round,
                eval_set=val_data,
                verbose_eval=100,
                early_stopping_rounds=50,
            )

            oof[val_idx] = [i[1] for i in clf.predict(X_train.iloc[val_idx][features], prediction_type='Probability')]
            predictions[:] += [i[1] / folds for i in clf.predict(X_test[features], prediction_type='Probability')]

            feat_imp_df['imp'] += clf.get_feature_importance() / folds
            if save_model:
                clf.save_model(f'model_{fold_}.json')

            del trn_data, val_data, clf
            gc.collect()
            print("fold n°{} finished. ".format(fold_))

    print("AUC score: {}".format(roc_auc_score(y, oof)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof])))

    return feat_imp_df, oof, predictions

feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y, model='xgb', seed=2022)
test['predict'] = predictions_lgb
test[['predict']].to_csv('../prediction_result/result.csv', index=False)