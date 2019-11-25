import pandas as pd
import numpy as np
import os
os.chdir('C:/Users/ALLEN/Desktop/PythonTest/HousePrice/')
import preprocess as pre
import model
train = pd.read_csv('C:/Users/ALLEN/Desktop/PythonTest/HousePrice/train.csv')
test = pd.read_csv('C:/Users/ALLEN/Desktop/PythonTest/HousePrice/test.csv')
x_train, y_train,test,test_ID = pre.input_data(train,test)
model.run_model(x_train,y_train,test,test_ID,'nn')


