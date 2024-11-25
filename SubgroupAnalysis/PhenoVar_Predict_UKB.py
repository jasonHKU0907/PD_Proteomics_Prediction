

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
outfile = dpath + 'Neurology_Revision/Results/SubGroupAnalysis/Pred_UKB.csv'
pro_f_lst1 = ['NEFL', 'ITGAV', 'HPGDS', 'HNMT', 'TPPP3', 'EDA2R', 'LXN', 'IL13RA1', 'BAG3', 'WARS', 'SCG2']
pro_f_lst2 = ['SCARF2', 'MERTK', 'CXCL9', 'TNXB', 'KLK8', 'NCAM1', 'CDH15', 'ACE2', 'TNFSF13', 'XG', 'NPPB']

cov_f_lst = ['DM_AGE', 'DM_GENDER', 'DM_ETH', 'DM_TDI', 'DM_EDUC',
              'LS_ALC', 'LS_SMK', 'LS_PA', 'LS_DIET', 'LS_SOC', 'LS_SLP', 'LS_WP',
              'PM_BMI', 'PM_BMR', 'PM_WT', 'PM_HT', 'PM_WC', 'PM_DBP', 'PM_SBP', 'PM_HGS', 'PM_PR',
              'BF_LP_TC', 'BF_LP_HDLC', 'BF_LP_LDLC', 'BF_LP_TRG',
              'BF_GL_GL', 'BF_GL_HBA1C', 'BF_IF_CRP',
              'BF_EC_IGF1', 'BF_EC_SHBG',
              'BF_LV_AP', 'BF_LV_ALT', 'BF_LV_AST', 'BF_LV_GGT',
              'BF_RE_ALB', 'BF_RE_TPRO', 'BF_RE_CR', 'BF_RE_CYSC', 'BF_RE_UREA',
              'BF_BC_WBC', 'BF_BC_RBC', 'BF_BC_HB', 'BF_BC_HCT', 'BF_BC_MCV', 'BF_BC_MCH', 'BF_BC_PLT',
             'SLP_DUR', 'SLP_GUM', 'SLP_MEP', 'SLP_NDD', 'SLP_INS', 'SLP_SNOR', 'SLP_DAYD', 'SLP_DIS',
             'x1920_0_0', 'x1930_0_0', 'x1940_0_0', 'x1950_0_0', 'x1960_0_0', 'x1970_0_0',
             'x1980_0_0', 'x1990_0_0', 'x2000_0_0', 'x2010_0_0', 'x2020_0_0', 'x2030_0_0',
             'x2040_0_0', 'x2050_0_0', 'x2060_0_0', 'x2070_0_0', 'x2080_0_0', 'x20127_0_0',
             'RF_BetaB', 'RF_CalCB', 'RF_TBI', 'RF_NSAIDs', 'RF_AGR',
             'RF_Coffee', 'RF_Water', 'RF_Urate', 'RF_Rural',
             'PS_DEP', 'PS_RBD', 'PS_UI', 'PS_ED', 'PS_CSP', 'PS_ANX', 'PS_OH', 'PS_HYM',
             'URI_MROALB', 'URI_CRE', 'URI_POS', 'URI_SOD']

target_df = pd.read_csv(dpath + 'Revision_R2/Data/TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Revision_R2/Data/Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)
pro_df = pd.read_csv(dpath + 'Revision_R2/Data/Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst1 + pro_f_lst2)

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'inner', on = ['eid'])

fold_id_lst = [i for i in range(10)]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}


pred_df1 = get_pred_probs(cov_f_lst, mydf, fold_id_lst, my_params, 'cov_all')
pred_df2 = get_pred_probs(['DM_AGE', 'DM_EDUC', 'RF_TBI', 'BF_RE_CR'], mydf, fold_id_lst, my_params, 'cov_4v')
pred_df3 = get_pred_probs(pro_f_lst1+pro_f_lst2, mydf, fold_id_lst, my_params, 'ProPanel')
pred_df4 = get_pred_probs(pro_f_lst1, mydf, fold_id_lst, my_params, 'pro_11v')
pred_df5 = get_pred_probs(['DM_AGE', 'DM_EDUC', 'RF_TBI', 'BF_RE_CR'] + pro_f_lst1, mydf, fold_id_lst, my_params, 'PDRP')
myout_df = pd.merge(mydf[['eid', 'target_y', 'BL2Target_yrs', 'Region_code', 'DM_AGE', 'DM_GENDER']], pred_df1[['eid', 'y_pred_cov_all']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df2[['eid', 'y_pred_cov_4v']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df3[['eid', 'y_pred_ProPanel']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df4[['eid', 'y_pred_pro_11v']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df5[['eid', 'y_pred_PDRP']], how = 'inner', on = ['eid'])
myout_df.to_csv(outfile, index = False)

roc_auc_score(myout_df.target_y, myout_df.y_pred_cov_all)
roc_auc_score(myout_df.target_y, myout_df.y_pred_cov_select)
roc_auc_score(myout_df.target_y, myout_df.y_pred_pro_all)
roc_auc_score(myout_df.target_y, myout_df.y_pred_pro_select)


