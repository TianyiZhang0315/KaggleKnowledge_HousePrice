from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, LeakyReLU,BatchNormalization
def run_model(train,y_train,test,test_ID,model='nn'):
    if model == 'nn':
        model = simple_nn(train,y_train,test)
        y_pred = np.expm1(model.predict(test))
    elif model == 'average':
        y_pred = average_model(train,y_train,test)
    
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = y_pred
    sub.to_csv('submission.csv',index=False)
    
    
def simple_nn(train,y_train,test):
    
    n_hidden_1 = 128 
    n_input = train.shape[1]
    n_classes = 1 
    training_epochs = 3000 
    batch_size = 32 
    model = Sequential()
    
    model.add(Dense(n_hidden_1,input_shape=(n_input,)))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.05))
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.05))
    #model.add(Dropout(rate = 0.3))
    model.add(Dense(n_classes))
    import keras.backend as K
    
    def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    history = model.fit(train, y_train, batch_size=batch_size, shuffle = True, epochs=training_epochs, validation_split=0.1)
    #plot epoch-rmse
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    

    return model
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def average_model(train,y_train,test):
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    print('xgb model rmse:{}'.format(rmsle(y_train,xgb_train_pred)))
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test))
    print('lgb model rmse:{}'.format(rmsle(y_train,lgb_train_pred)))
    ensemble = xgb_pred*0.5 + lgb_pred*0.5
    return ensemble
