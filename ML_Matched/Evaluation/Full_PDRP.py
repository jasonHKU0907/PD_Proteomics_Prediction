

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

def get_pred_probs(tmp_f, mydf, fold_id_lst, my_params, col_name):
    eid_lst, region_lst = [], []
    y_test_lst, y_pred_lst = [], []
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
        my_lgb.set_params(**my_params)
        calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
        calibrate.fit(X_train, y_train)
        y_pred_prob = calibrate.predict_proba(X_test)[:, 1].tolist()
        y_pred_lst += y_pred_prob
        y_test_lst += mydf.target_y.iloc[test_idx].tolist()
        eid_lst += mydf.eid.iloc[test_idx].tolist()
        region_lst += mydf.Region_code.iloc[test_idx].tolist()
    myout_df = pd.DataFrame([eid_lst, region_lst, y_test_lst, y_pred_lst]).T
    myout_df.columns = ['eid', 'Region_code', 'target_y', 'y_pred_'+col_name]
    myout_df[['eid', 'Region_code']] = myout_df[['eid', 'Region_code']].astype('int')
    return myout_df

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'
outputfile = outpath + 'Matched/PDRP_Retrained_OldFeat_PredProbs.csv'

pro_f_df = pd.read_csv(outpath + 'UKB_ALL/FULL/s2_AccAUC_TotalGain.csv')
nb_top_pros = get_top_pros(pro_f_df)
imp_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_f_lst = [f for f in imp_f_lst if '_' not in f]
cov_f_lst = [f for f in imp_f_lst if f not in pro_f_lst]

target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code']+cov_f_lst)
pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)
match_df = pd.read_csv(dpath + 'matchit_nearest_data_one_to_one.csv')

mydf = pd.merge(match_df, target_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, pro_df, how = 'left', on = ['eid'])

fold_id_lst = [i for i in range(10)]

my_params = {'n_estimators': 300,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

pred_df = get_pred_probs(imp_f_lst, mydf, fold_id_lst, my_params, 'probs')

myout_df = pd.merge(target_df[['eid', 'BL2Target_yrs']], pred_df, how = 'inner', on = ['eid'])

myout_df.to_csv(outputfile, index = False)

roc_auc_score(myout_df.target_y, myout_df.y_pred_probs)



