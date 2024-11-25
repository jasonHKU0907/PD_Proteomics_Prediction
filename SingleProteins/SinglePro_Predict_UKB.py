
from tqdm import tqdm
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

def get_pred_probs(tmp_f, mydf, fold_id_lst, my_params, col_name):
    eid_lst, region_lst = [], []
    y_test_lst, y_pred_lst = [], []
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True,  n_jobs=4, verbosity=-1, seed=2020)
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

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outfile = dpath + 'Neurology_Revision/Results/SingleProteins/Pred_UKB.csv'
pro_f_lst = ['NEFL', 'ITGAV', 'HPGDS', 'HNMT', 'TPPP3', 'EDA2R', 'LXN', 'IL13RA1', 'BAG3', 'WARS', 'SCG2',
             'SCARF2', 'MERTK', 'CXCL9', 'TNXB', 'KLK8', 'NCAM1', 'CDH15', 'ACE2', 'TNFSF13', 'XG', 'NPPB']

cov_f_lst = ['DM_AGE', 'DM_EDUC', 'RF_TBI', 'BF_RE_CR']

target_df = pd.read_csv(dpath + 'Revision_R2/Data/TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Revision_R2/Data/Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code', 'DM_GENDER'] + cov_f_lst)
pro_df = pd.read_csv(dpath + 'Revision_R2/Data/Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'inner', on = ['eid'])

fold_id_lst = [i for i in range(10)]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}


cov_pred_df = get_pred_probs(cov_f_lst, mydf, fold_id_lst, my_params, 'cov_4v')
myout_df = pd.merge(mydf[['eid', 'target_y', 'BL2Target_yrs', 'Region_code', 'DM_AGE', 'DM_GENDER']], cov_pred_df[['eid', 'y_pred_cov_4v']], how='inner', on=['eid'])

for f in tqdm(pro_f_lst):
    tmpdf = mydf[['eid', f]]
    myout_df = pd.merge(myout_df, tmpdf, how = 'inner', on = ['eid'])
    tmp_pred_f = get_pred_probs(cov_f_lst+ [f], mydf, fold_id_lst, my_params, 'cov_'+f)
    myout_df = pd.merge(myout_df, tmp_pred_f[['eid', 'y_pred_cov_' + f]], how = 'inner', on = ['eid'])

myout_df.to_csv(outfile, index = False)

roc_auc_score(myout_df.target_y, myout_df.y_pred_cov_all)
roc_auc_score(myout_df.target_y, myout_df.y_pred_cov_select)
roc_auc_score(myout_df.target_y, myout_df.y_pred_pro_all)
roc_auc_score(myout_df.target_y, myout_df.y_pred_pro_select)

