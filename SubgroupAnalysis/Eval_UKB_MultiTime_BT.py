
# Calculate the AUC based on predicted probabilities derived from ALL incidence PD without re-training

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score, average_precision_score
from Utility.Training_Utilities import *
import random
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    apr = average_precision_score(y_test, pred_prob)
    brier = brier_score_loss(y_test, pred_prob)
    nnd = 1 / Youden
    evaluations = np.round((cutoff, acc, sens, spec, prec, Youden, f1, auc, apr, nnd, brier), 4)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND', 'BRIER']
    return evaluations

def get_avg_output(mydf, gt_col, pred_col, cutoff, nb_iters):
    idx_lst = [ele for ele in range(len(mydf))]
    out_df = pd.DataFrame()
    for i in range(nb_iters):
        random.seed(i)
        bt_idx = [random.choice(idx_lst) for _ in range(len(idx_lst))]
        mydf_bt = mydf.copy()
        mydf_bt = mydf_bt.iloc[bt_idx, :]
        tmpout_df = get_eval(mydf_bt[gt_col], mydf_bt[pred_col], cutoff)
        out_df = pd.concat([out_df, tmpout_df], axis = 0)
    result_df = out_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['STD'] = result_df.std(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)
    output_lst = []
    for i in range(11):
        my_mean = str(np.round(result_df['Median'][i], 3))
        #my_std = str(np.round(result_df['STD'][i], 3))
        my_lbd = str(np.round(result_df['LBD'][i], 3))
        my_ubd = str(np.round(result_df['UBD'][i], 3))
        output_lst.append(my_mean + ' [' + my_lbd + '-'+my_ubd + ']')
    result_df['output'] = output_lst
    return result_df.T

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outfile = dpath + 'Neurology_Revision/Results/SubGroupAnalysis/EVAL_UKB_MultiTime_BT.csv'

mydf = pd.read_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/Pred_UKB.csv')
pred_f_lst = mydf.columns.tolist()[6:]


fold_id_lst = [i for i in range(10)]
myout_df = pd.DataFrame()

for yr in [2, 4, 6, 8, 10, 12, 14, 100]:
    tmpdf = mydf.copy()
    tmpdf['target_y'].loc[tmpdf.BL2Target_yrs >= yr] = 0
    tmpout_df = pd.DataFrame()
    for f in pred_f_lst:
        opt_ct = Find_Optimal_Cutoff(mydf.target_y, mydf[f])[0]
        result_df = get_avg_output(mydf, 'target_y', f, opt_ct, nb_iters=1000).iloc[-1, :]
        tmpout_df = pd.concat([tmpout_df, result_df], axis=1)
    tmpout_df.columns = [f + '_' + str(yr) for f in pred_f_lst]
    myout_df = pd.concat([myout_df, tmpout_df], axis = 1)

myout_df = myout_df.T
myout_df.to_csv(outfile, index=True)

