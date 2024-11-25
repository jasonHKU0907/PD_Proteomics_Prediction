
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

fold_id_lst = [i for i in range(10)]



dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'
#outfile = dpath + 'MultiObsTime/ProPANEL.csv'
outfile = dpath + 'MultiObsTime/FULL.csv'
#outfile = dpath + 'MultiObsTime/wo3variables.csv'
#outfile = dpath + 'MultiObsTime/Statistics_RM_IL13RA1.csv'

out_lst = []
for yr in [2, 4, 6, 8, 10, 12, 14, 100]:
    #mydf = pd.read_csv(dpath + 'UKB_ALL/PRO_PANEL/s6_ProPANEL_PredProbs.csv')
    mydf = pd.read_csv(dpath + 'UKB_ALL/FULL/s3_PDRP_PredProbs.csv')
    #mydf = pd.read_csv(dpath + 'MultiObsTime/PredProbs_RM_IL13RA1.csv')
    mydf['target_y'].loc[mydf.BL2Target_yrs >= yr] = 0
    opt_ct = Find_Optimal_Cutoff(mydf.target_y, mydf.y_pred_probs)[0]
    opt_ct = np.round(opt_ct, 3)
    out_lst.append(get_avg_output(mydf, 'target_y', 'y_pred_probs', 'Region_code', fold_id_lst, opt_ct).iloc[12,:])
    print(mydf['target_y'].sum())


out_df = pd.DataFrame(out_lst)
out_df.index = [2, 4, 6, 8, 10, 12, 14, 100]
out_df.to_csv(outfile, index=True)


