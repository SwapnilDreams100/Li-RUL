from scipy.io import loadmat, whosmat
import numpy as np
import pandas as pd
from pyswarm import pso
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import gc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import json
import os
import math
import random
random.seed(42)
np.random.seed(42)

from plotting_utils import *
from kernels import *

def generate_data(discharge_data):
    data = {}
    for cycle in discharge_data.keys():
        data[str(cycle)] = {}
        test_list = discharge_data[cycle]["time"]
        i_500 = next(x for x, val in enumerate(test_list) if val > 500.0) 
        i_1000 = next(x for x, val in enumerate(test_list) if val > 1000.0) 
        i_1500 = next(x for x, val in enumerate(test_list) if val > 1500.0) 
        i_2000 = next(x for x, val in enumerate(test_list) if val > 2000.0) 
        i_2500 = next(x for x, val in enumerate(test_list) if val > 2500.0) 

        data[str(cycle)]['v_500'] = discharge_data[cycle]["voltage_battery"][i_500]
        data[str(cycle)]['v_1000'] = discharge_data[cycle]["voltage_battery"][i_1000]
        data[str(cycle)]['v_1500'] = discharge_data[cycle]["voltage_battery"][i_1500]
        data[str(cycle)]['v_2000'] = discharge_data[cycle]["voltage_battery"][i_2000]
        data[str(cycle)]['v_2500'] = discharge_data[cycle]["voltage_battery"][i_2500]

#         data[str(cycle)]['r_500'] = discharge_data[cycle]["voltage_battery"][i_500]/discharge_data[cycle]["current_battery"][i_500]
#         data[str(cycle)]['r_1000'] = discharge_data[cycle]["voltage_battery"][i_1000]/discharge_data[cycle]["current_battery"][i_1000]
#         data[str(cycle)]['r_1500'] = discharge_data[cycle]["voltage_battery"][i_1500]/discharge_data[cycle]["current_battery"][i_1500]
#         data[str(cycle)]['r_2000'] = discharge_data[cycle]["voltage_battery"][i_2000]/discharge_data[cycle]["current_battery"][i_2000]
#         data[str(cycle)]['r_2500'] = discharge_data[cycle]["voltage_battery"][i_2500]/discharge_data[cycle]["current_battery"][i_2500]

#         data[str(cycle)]['vv_500'] = discharge_data[cycle]["voltage_load"][i_500]
#         data[str(cycle)]['vv_1000'] = discharge_data[cycle]["voltage_load"][i_1000]
#         data[str(cycle)]['vv_1500'] = discharge_data[cycle]["voltage_load"][i_1500]
#         data[str(cycle)]['vv_2000'] = discharge_data[cycle]["voltage_load"][i_2000]
#         data[str(cycle)]['vv_2500'] = discharge_data[cycle]["voltage_load"][i_2500]

        data[str(cycle)]['capacity'] = discharge_data[cycle]["capacity"][0]
        
    return data

path = './data_json/'
battery_list = ['B0005','B0006','B0007','B0018']
data_list = []
params_list ={}

for battery in battery_list:
    with open(path+battery+'_discharge.json') as f:    
        discharge_data = json.load(f)
    data = generate_data(discharge_data)   
    df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)
    df['capacity2'] = df['capacity'] - df['capacity'].shift(1)
    df['capacity2'] = df['capacity2'].fillna(method = 'bfill')
    data_list.append(df)
    
def get_params():
    global params_list

    with open('2_params.json', 'r') as f:
        params_list = json.load(f)

def save_txt(content, filename):
    global params_list, current_battery

    k = current_battery +'_'+ filename

    if k in params_list.keys():
        params_list[k].update(content)
    else:
        params_list[k] = content

    with open('2_params.json', 'w') as f:
        json.dump(params_list, f)
    
def applyPSO(lb, ub, model):
    
    xopt, _ = pso(model, lb, ub, maxiter=3, swarmsize=100, debug=True)
    return(xopt)

def create_model(model_method, params, tr, tr_labels, tt, tt_labels, pred=False, save =False, inf = False):
    
    if model_method == 'gbt':
        p_list = ['nest','lr']
        model = XGBRegressor(random_state=42, max_depth=5, learning_rate=params[1], n_estimators= int(params[0]))
        model.fit(tr, tr_labels)
        preds = model.predict(tt)
        
    elif model_method == 'rsvr':
        p_list = ['C', 'epsilon']
        model = SVR(C=params[0], epsilon = params[1], kernel = 'rbf', max_iter=100000)
        model.fit(tr, tr_labels)
        preds = model.predict(tt)
    
    elif model_method == 'msvr': 
        
        p_list = ['C', 'epsilon', 'beta']
        gamma = 1 / (len(tr.columns) * np.array(tr).var())   

        beta = params[2]
        kernels = [rbf(gamma), polynomial(gamma, 0.0, 1)]
        betas = np.array([beta, 1.0-beta])
            
        def unique_kernel(X, y):
            return np.dot(betas, np.array([kernel(X, y) for kernel in kernels]))

        def gram_matrix(X1, X2):
            gram = np.zeros((X1.shape[0], X2.shape[0]))
            for n, x_n in enumerate(X1):
                for m, x_m in enumerate(X2):
                    x_n = x_n.flatten()
                    x_m = x_m.flatten()
                    gram[n,m] = unique_kernel(x_n, x_m)
            return gram
        
        model = SVR(C=params[0], epsilon = params[1], kernel = 'precomputed', max_iter=100000) # precomputed for custom kernel
        model.fit(gram_matrix(tr.values,tr.values),tr_labels)
        preds = model.predict( gram_matrix(tt.values, tr.values))
        
    elif model_method == 'rf':
        p_list = ['nest', 'max_depth']
        model = RandomForestRegressor(n_estimators = params[0], random_state = 42, max_depth = params[1])
        model.fit(tr, tr_labels)
        preds = model.predict(tt)
        
    elif model_method == 'lr':
        model = linear_model.LinearRegression(n_jobs = -1)
        model.fit(train_features, train_labels)
        preds = model.predict(test_features)
    
    else:
        print('error in type')
        return None    

    error = mean_squared_error(preds, tt_labels)
    if inf ==True:
        print('MSE:', error)

    if save ==True:
        param_dict = {}
        print('MSE: ',error)
        for i,j in zip(p_list, params): 
            param_dict[i] = j
        save_txt(param_dict, model_method)
    
    if pred == True:
        return preds
    
    return error

def run_rsvr_kfold(params):

    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

    y_pred = np.zeros((len(train_features), ))
    oof_pred = np.zeros((len(train_features), ))
    
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train_features, train_labels)):
        
        x_train, x_val = train_features.iloc[tr_ind], train_features.iloc[val_ind]
        y_train, y_val = train_labels[tr_ind], train_labels[val_ind]
        
        oof_pred[val_ind] = np.array(create_model('rsvr', params, x_train, y_train, x_val, y_val, pred=True))
        y_pred[val_ind] = np.array(y_val)
        
    error = mean_squared_error(oof_pred,y_pred)
    print(error)
    return error

def run_msvr_kfold(params):

    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

    y_pred = np.zeros((len(train_features), ))
    oof_pred = np.zeros((len(train_features), ))
    
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train_features, train_labels)):
        
        x_train, x_val = train_features.iloc[tr_ind], train_features.iloc[val_ind]
        y_train, y_val = train_labels[tr_ind], train_labels[val_ind]
        
        oof_pred[val_ind] = np.array(create_model('msvr', params, x_train, y_train, x_val, y_val, pred=True))
        y_pred[val_ind] = np.array(y_val)
        
    error = mean_squared_error(oof_pred,y_pred)
    print(error)
    return error

def run_gbt_kfold(params):

    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

    y_pred = np.zeros((len(train_features), ))
    oof_pred = np.zeros((len(train_features), ))
    
    for fold, (tr_ind, val_ind) in enumerate(kf.split(train_features, train_labels)):
        
        x_train, x_val = train_features.iloc[tr_ind], train_features.iloc[val_ind]
        y_train, y_val = train_labels[tr_ind], train_labels[val_ind]
        
        oof_pred[val_ind] = np.array(create_model('msvr', params, x_train, y_train, x_val, y_val, pred=True))
        y_pred[val_ind] = np.array(y_val)
        
    error = mean_squared_error(oof_pred,y_pred)
    print(error)
    return error

def run(train, test, batt_no):
    
    global train_features, test_features, train_labels, test_labels, tree_train_labels, tree_test_labels, current_battery

    current_battery = batt_no
    
    train_features = train.drop(['capacity', 'capacity2'],axis = 1)
    test_features = test.drop(['capacity','capacity2'],axis = 1)

    train_labels = train['capacity']
    test_labels = test['capacity']

    # For the tree based models
    tree_train_labels = train['capacity2']
    tree_test_labels = test['capacity2']
    
    # MSVM
    lb = [1.0, 0.01, 0.0]
    ub = [10.0, 0.1, 1.0]

    a = applyPSO(lb, ub, run_msvr_kfold, ['C', 'epsilon', 'beta'], batt_no)
    s_mult = create_model('msvr',list(a.values()), train_features, train_labels, test_features, test_labels, pred = True, save = True)
    
    gc.collect()
    
    # RSVR
    lb = [0.01, 0.0001]
    ub = [1000.0, 0.01]

    b = applyPSO(lb, ub, run_rsvr_kfold, ['C', 'epsilon'], batt_no)
    s_rbf = create_model('rsvr', list(b.values()), train_features, train_labels, test_features, test_labels, pred = True, save = True)
    
    gc.collect()
    
    #GBT
    lb = [0.01, 1000]
    ub = [0.5, 2000]

    d = applyPSO(lb, ub, run_gbt_kfold, ['lr','n_estimator'], batt_no)
    gbt_pred = create_model('gbt',list(d.values()), train_features, tree_train_labels, test_features, tree_test_labels, save = True)
  
    create_model('lr', [], train_features, train_labels, test_features, test_labels)
    
    e = [20, 10]
    rf_pred = create_model('rf',list(e.values()), train_features, tree_train_labels, test_features, tree_test_labels, save = True)
    
    plot_2([list(s_mult), s_rbf, list(test_labels)], batt_no,  list(pd.concat([train_labels, test_labels], axis = 0)))
    gc.collect()

def inference(train, test, batt_no, a, b, d, e):
    
    global train_features, test_features, train_labels, test_labels, tree_train_labels, tree_test_labels, current_battery
    
    current_battery = batt_no

    train_features = train.drop(['capacity', 'capacity2'],axis = 1)
    test_features = test.drop(['capacity','capacity2'],axis = 1)

    train_labels = train['capacity']
    test_labels = test['capacity']

    # For the tree based models
    tree_train_labels = train['capacity2']
    tree_test_labels = test['capacity2']
    
    # Multi-SVR
    s_mult = create_model('msvr',list(a.values()), train_features, train_labels, test_features, test_labels, pred = True, inf = True)
    
    gc.collect()

    #RSVR
    s_rbf = create_model('rsvr', list(b.values()), train_features, train_labels, test_features, test_labels, pred = True, inf = True)
    
    gc.collect()
    
    #GBT
    gbt_pred = create_model('gbt',list(d.values()), train_features, tree_train_labels, test_features, tree_test_labels, pred = True, inf = True)
  
    create_model('lr', [], train_features, train_labels, test_features, test_labels)
    
    #RF
    rf_pred = create_model('rf',list(e.values()), train_features, tree_train_labels, test_features, tree_test_labels, pred = True, inf = True)
    
    plot_2([list(s_mult), s_rbf, list(test_labels)], batt_no,  list(pd.concat([train_labels, test_labels], axis = 0)))

get_params()
if params_list.keys() == {}:
    print('no hyperparams, run with PSO model')

df_new1 = pd.concat([data_list[1], data_list[2], data_list[3]], axis = 0, ignore_index=True).sample(frac=1).reset_index(drop=True)

df_new2 = pd.concat([data_list[0], data_list[2], data_list[3]], axis = 0, ignore_index=True).sample(frac=1).reset_index(drop=True)

df_new3 = pd.concat([data_list[0], data_list[1], data_list[3]], axis = 0, ignore_index=True).sample(frac=1).reset_index(drop=True)

df_new4 = pd.concat([data_list[0], data_list[1], data_list[2]], axis = 0, ignore_index=True).sample(frac=1).reset_index(drop=True)

def main(run = False):
    global params_list
    if run == False:
        models = ['msvr', 'rsvr', 'gbt', 'rf']

        batt = '5'
        gc.collect()
        a = params_list[batt+'_'+models[0]]
        b = params_list[batt+'_'+models[1]]
        d = params_list[batt+'_'+models[2]]
        e = params_list[batt+'_'+models[3]]
        inference(df_new1, data_list[0], batt, a, b, d, e)
        print('done1\n')

        batt = '6'
        gc.collect()
        a = params_list[batt+'_'+models[0]]
        b = params_list[batt+'_'+models[1]]
        d = params_list[batt+'_'+models[2]]
        e = params_list[batt+'_'+models[3]]
        inference(df_new2, data_list[1], batt, a, b, d, e)
        print('done2\n')

        batt = '7'
        gc.collect()
        a = params_list[batt+'_'+models[0]]
        b = params_list[batt+'_'+models[1]]
        d = params_list[batt+'_'+models[2]]
        e = params_list[batt+'_'+models[3]]
        inference(df_new3, data_list[2], batt, a, b, d, e)
        print('done3\n')

        batt = '18'
        gc.collect()
        a = params_list[batt+'_'+models[0]]
        b = params_list[batt+'_'+models[1]]
        d = params_list[batt+'_'+models[2]]
        e = params_list[batt+'_'+models[3]]
        inference(df_new4, data_list[3], batt, a, b, d, e)
        print('done4\n')
    else:
        models = ['msvr', 'rsvr', 'gbt', 'rf']

        
        batt = '5'
        gc.collect()
        run(df_new1, data_list[0], batt)
        print('done1\n')

        batt = '6'
        gc.collect()
        run(df_new2, data_list[1], batt)
        print('done2\n')

        batt = '7'
        gc.collect()
        run(df_new3, data_list[2], batt)
        print('done3\n')

        batt = '18'
        gc.collect()
        run(df_new4, data_list[3], batt)
        print('done4\n')
main()