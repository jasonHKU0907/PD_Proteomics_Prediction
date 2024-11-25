

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Neurology_Revision/Results/RandomCV/'
outfile = outpath + 'ProPanelPrediction.csv'

pro_f_df = pd.read_csv(dpath + 'Revision_R2/Results/UKB_ALL/PRO_PANEL/s5_AccAUC_TotalGain.csv')
pro_f_lst = pro_f_df.Pro_code.tolist()[:22]
pro_df = pd.read_csv(dpath + 'Data/Proteomics/Raw/ProteomicsData.csv', usecols = ['eid']+pro_f_lst)
target_df = pd.read_csv(dpath + 'Data/TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Data/Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])


my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

mykf = StratifiedKFold(n_splits = 10, random_state = 2020, shuffle = True)

eid_lst, region_lst = [], []
y_test_lst, y_pred_lst = [], []
i=0
for train_idx, test_idx in mykf.split(mydf[pro_f_lst], mydf.target_y):
    X_train, X_test = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[test_idx][pro_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=10)
    calibrate.fit(X_train, y_train)
    y_pred_prob = calibrate.predict_proba(X_test)[:, 1].tolist()
    y_pred_lst += y_pred_prob
    y_test_lst += mydf.target_y.iloc[test_idx].tolist()
    eid_lst += mydf.eid.iloc[test_idx].tolist()
    region_lst += [i]*len(test_idx)
    i+=1


myout_df = pd.DataFrame([eid_lst, region_lst, y_test_lst, y_pred_lst]).T
myout_df.columns = ['eid', 'Fold_code', 'target_y', 'y_pred_probs']
myout_df[['eid', 'Fold_code']] = myout_df[['eid', 'Fold_code']].astype('int')

myout_df = pd.merge(target_df[['eid', 'BL2Target_yrs']], myout_df, how = 'inner', on = ['eid'])
myout_df.to_csv(outfile, index = False)

roc_auc_score(myout_df.target_y, myout_df.y_pred_probs)

