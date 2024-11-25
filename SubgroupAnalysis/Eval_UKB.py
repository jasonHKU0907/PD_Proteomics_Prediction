
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

def get_avg_output(mydf, y_true_col, y_pred_col, fold_col, fold_id_lst, cutoff):
    result_df = pd.DataFrame()
    for fold_id in fold_id_lst:
        tmp_idx = mydf[fold_col].index[mydf[fold_col] == fold_id]
        tmpdf = mydf.iloc[tmp_idx]
        tmpdf.reset_index(inplace = True, drop = True)
        y_test, pred_prob = tmpdf[y_true_col], tmpdf[y_pred_col]
        tmp_result_df = get_eval(y_test, pred_prob, cutoff)
        result_df = pd.concat([result_df, tmp_result_df], axis = 0)
    result_df = result_df.T
    result_df['MEAN'] = result_df.mean(axis=1)
    result_df['STD'] = result_df.std(axis=1)
    output_lst = []
    for i in range(11):
        my_mean = str(np.round(result_df['MEAN'][i], 3))
        my_std = str(np.round(result_df['STD'][i], 3))
        output_lst.append(my_mean + 'X' + my_std)
    result_df['output'] = output_lst
    return result_df.T

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outfile = dpath + 'Neurology_Revision/Results/SubGroupAnalysis/EVAL_UKB.csv'

mydf = pd.read_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/Pred_UKB.csv')
pred_f_lst = mydf.columns.tolist()[6:]


fold_id_lst = [i for i in range(10)]
myout_df_all, myout_df_female, myout_df_male, myout_df_young60 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
myout_df_old60, myout_df_young65, myout_df_old65 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


mydf_male = mydf.loc[mydf.DM_GENDER == 1]
mydf_male.reset_index(inplace = True, drop = True)

mydf_female = mydf.loc[mydf.DM_GENDER == 0]
mydf_female.reset_index(inplace = True, drop = True)

mydf_young60 = mydf.loc[mydf.DM_AGE<60]
mydf_young60.reset_index(inplace = True, drop = True)

mydf_old60 = mydf.loc[mydf.DM_AGE>=60]
mydf_old60.reset_index(inplace = True, drop = True)

mydf_young65 = mydf.loc[mydf.DM_AGE<=65]
mydf_young65.reset_index(inplace = True, drop = True)

mydf_old65 = mydf.loc[mydf.DM_AGE>=66]
mydf_old65.reset_index(inplace = True, drop = True)

for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf.target_y, mydf[f])[0]
    result_df = get_avg_output(mydf, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_all = pd.concat([myout_df_all, result_df], axis = 1)

myout_df_all.columns = pred_f_lst


for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_female.target_y, mydf_female[f])[0]
    result_df = get_avg_output(mydf_female, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_female = pd.concat([myout_df_female, result_df], axis = 1)

myout_df_female.columns = [f + '_female' for f in pred_f_lst]


for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_male.target_y, mydf_male[f])[0]
    result_df = get_avg_output(mydf_male, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_male = pd.concat([myout_df_male, result_df], axis = 1)

myout_df_male.columns = [f + '_male' for f in pred_f_lst]


for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_young60.target_y, mydf_young60[f])[0]
    result_df = get_avg_output(mydf_young60, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_young60 = pd.concat([myout_df_young60, result_df], axis = 1)

myout_df_young60.columns = [f + '_young60' for f in pred_f_lst]


for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_old60.target_y, mydf_old60[f])[0]
    result_df = get_avg_output(mydf_old60, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_old60 = pd.concat([myout_df_old60, result_df], axis = 1)

myout_df_old60.columns = [f + '_old60' for f in pred_f_lst]


for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_young65.target_y, mydf_young65[f])[0]
    result_df = get_avg_output(mydf_young65, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_young65 = pd.concat([myout_df_young65, result_df], axis = 1)

myout_df_young65.columns = [f + '_young65' for f in pred_f_lst]



for f in pred_f_lst:
    opt_ct = Find_Optimal_Cutoff(mydf_old65.target_y, mydf_old65[f])[0]
    result_df = get_avg_output(mydf_old65, 'target_y', f, 'Region_code', fold_id_lst, opt_ct).iloc[-1,:]
    myout_df_old65 = pd.concat([myout_df_old65, result_df], axis = 1)

myout_df_old65.columns = [f + '_old65' for f in pred_f_lst]

myout_df = pd.concat([myout_df_all, myout_df_female, myout_df_male, myout_df_young60, myout_df_old60, myout_df_young65, myout_df_old65], axis = 1)
myout_df = myout_df.T

myout_df.to_csv(outfile)

