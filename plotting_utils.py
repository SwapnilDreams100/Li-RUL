import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_1(preds_list, batt_no, tr_labels):
    files = ['MSVR','RSVR','PSVR']
    c = ['b','c','r']
    fig=plt.figure()
    for x, col, file in zip(preds_list[:-1], c, files):
        s = np.add(np.arange(len(x)), (len(tr_labels)-len(preds_list[-1])) * np.ones(len(x)))
        plt.plot(s, x, color = col, label = file )
    plt.plot(np.arange(len(tr_labels)), tr_labels, color= 'k', label= 'original')
    plt.legend(loc='upper right', frameon=False)
    plt.figure(figsize=(300,300))
    fig.savefig('./figs/1_'+str(batt_no)+'.jpg')

def plot_2(preds_list, batt_no, tr_labels):
    files = ['MSVR','RSVR']
    c = ['b','r']
    fig=plt.figure()
    for x, col, file in zip(preds_list[:-1], c, files):
        s = np.add(np.arange(len(x)), (len(tr_labels)-len(preds_list[-1])) * np.ones(len(x)))
        plt.plot(s, x, color = col, label = file )
    plt.plot(np.arange(len(tr_labels)), tr_labels, color= 'k', label= 'original')
    plt.legend(loc='upper right', frameon=False)
    plt.figure(figsize=(300,300))
    # fig.savefig('./figs/2_'+str(batt_no)+'.jpg')