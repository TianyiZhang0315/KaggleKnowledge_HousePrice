# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:22:20 2019

@author: ALLEN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics



def input_data(train,test):
    #save id column
    train.drop('Id',axis = 1, inplace = True)
    test_id = test['Id'] 
    test.drop('Id',axis = 1, inplace = True)
    #detect outliers
#    fig, ax = plt.subplots()
#    ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
#    plt.ylabel('SalePrice', fontsize=13)
#    plt.xlabel('GrLivArea', fontsize=13)
#    plt.show()
    #drop outliers
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#    fig, ax = plt.subplots()
#    ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
#    plt.ylabel('SalePrice', fontsize=13)
#    plt.xlabel('GrLivArea', fontsize=13)
#    plt.show()
    
    #check label distribution
    #sns.distplot(train['SalePrice'] , fit=norm)
    #log transformation
    train["SalePrice"] = np.log1p(train["SalePrice"])
    #sns.distplot(train['SalePrice'] , fit=norm)
    
    #concate
    ntrain = train.shape[0]
    all_data = pd.concat((train,test),sort=False).reset_index(drop = True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    y_train = train['SalePrice'].to_numpy()
    #check missing value
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head(20)
    #data correlation
#    corrmat = train.corr()
#    plt.subplots(figsize=(12,9))
#    sns.heatmap(corrmat, vmax=0.9, square=True)
    #imputing missing value
    
    #check unique values
    #for item in all_data.columns:
    #    print(all_data[item].value_counts())
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    #check missing again
#    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
#    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
#    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#    missing_data.head()
    
    ####Transforming some numerical variables that are really categorical####
    
    #MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    
    
    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    
    
    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    
    #categorical feature encoding
    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))
    #add feature 'totalsf'
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    #check skewed features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    
    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(20)
    #sns.distplot(all_data['MiscVal'] , fit=norm)
    
    #Box Cox transformation(skewness > 0,75)
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.25
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
        
    #all_data[skewed_features] = np.log1p(all_data[skewed_features])
        
        
    # Check again
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(20)
    #sns.distplot(all_data['MiscVal'] , fit=norm)
    
    #get dummy
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)
    #separate train test
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    #to numpy
    train = train.to_numpy()
    test = test.to_numpy()
    return train, y_train, test,test_id