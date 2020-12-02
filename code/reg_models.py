import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU, PReLU
from keras.optimizers import SGD, Adam

# Ridge
class Model_Ridge:
    def __init__(self, params):
        self.params = params
        self.model = Ridge(**self.params)
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        #x = self.scaler.transform(x)
        #pred = self.model.predict(x)
        return self.model.predict(x)
    
# XGBM
class Model_Xgb:
    def __init__(self, params):
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'n_jobs' in params:
            params['n_jobs'] = int(params['n_jobs'])
        if 'max_delta_step' in params:
            params['max_delta_step'] = int(params['max_delta_step'])
        
        self.params = params
        #self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, tr_x, tr_y, va_x, va_y):
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        params = self.params
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist, verbose_eval=0)
    
    def predict(self, x):
        data = xgb.DMatrix(x)
        return self.model.predict(data)
    

# LightGBM
class Model_Lgbm:
    def __init__(self, params):
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'num_leaves' in params:
            params['num_leaves'] = int(params['num_leaves'])
        if 'min_data_in_leaf' in params:
            params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])
        if 'subsample_for_bin' in params:
            params['subsample_for_bin'] = int(params['subsample_for_bin'])
        if 'min_child_samples' in params:
            params['min_child_samples'] = int(params['min_child_samples'])
        if 'subsample_freq' in params:
            params['subsample_freq'] = int(params['subsample_freq'])
        if 'n_jobs' in params:
            params['n_jobs'] = int(params['n_jobs'])

        self.params = params
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        return self.model.predict(x)
    
# Random Forest
class Model_RF:    
    def __init__(self, params):
        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'min_samples_split' in params:
            params['min_samples_split'] = int(params['min_samples_split'])
        if 'min_samples_leaf' in params:
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
        
        self.params = params
        self.model = RandomForestRegressor(**self.params)
        
    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y)
    
    def predict(self, x):
        return self.model.predict(x)

# Extremely randomized trees
class Model_ET:
    def __init__(self, params):
        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'min_samples_split' in params:
            params['min_samples_split'] = int(params['min_samples_split'])
        if 'min_samples_leaf' in params:
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
        
        self.params = params
        self.model = ExtraTreesRegressor(**self.params)
        
    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y)
    
    def predict(self, x):
        return self.model.predict(x)
    
# Support Vector Machine
class Model_SVR:
    def __init__(self, params):
        self.params = params
        self.model = SVR(**self.params)

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y)
        #self.model = model
    
    def predict(self, x):
        return self.model.predict(x)

# Neural Network
class Model_NN:

    def __init__(self, params, input_x):
        self.params = params
    
        # Layer Setting
        self.model = Sequential()
        self.model.add(Dropout(self.params['input_dropout'], input_shape=(input_x.shape[1],)))# 入力層
        for i in range(int(self.params['hidden_layers'])):# 中間層
            self.model.add(Dense(int(self.params['hidden_units'])))
            if self.params['batch_norm'] == 'before_act':
                self.model.add(BatchNormalization())
            if self.params['hidden_activation'] == 'prelu':
                self.model.add(PReLU())
            elif self.params['hidden_activation'] == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(self.params['input_dropout']))
        self.model.add(Dense(1))# 出力層

        # オプティマイザ
        if self.params['optimizer']['type'] == 'sgd':
            optimizer = SGD(lr=self.params['optimizer']['lr'], decay=1e-6, momentum=0.9, nesterov=True)
        elif self.params['optimizer']['type'] == 'adam':
            optimizer = Adam(lr=self.params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer, metrics=['mse'])

    def fit(self, tr_x, tr_y, va_x, va_y):
        # 標準化
        self.scaler = StandardScaler()
        tr_x = self.scaler.fit_transform(tr_x)
        va_x = self.scaler.transform(va_x)        

        # エポック数、アーリーストッピング
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        nb_epoch = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        # 学習の実行
        history = self.model.fit(tr_x, tr_y,
                                 epochs=nb_epoch,
                                 batch_size=int(self.params['batch_size']), verbose=0,
                                 validation_data=(va_x, va_y),
                                 callbacks=[early_stopping])

    def predict(self, x):
        # 予測
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred
