

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision/Results/'
outfile = outpath + 'UKB_ALL_Matched/FULL/FeaImportance.csv'

pro_f_df = pd.read_csv(outpath + 'UKB_ALL_Matched/PRO_PANEL/AccAUC_TotalGain.csv')
nb_top_pros = get_top_pros(pro_f_df)
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
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


target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)
pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)
match_df = pd.read_csv(dpath + 'matchit_nearest_data_one_to_one.csv')

mydf = pd.merge(match_df, target_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, pro_df, how = 'left', on = ['eid'])


my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tmp_f_lst = pro_f_lst + cov_f_lst
tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(tmp_f_lst))
fold_id_lst = [i for i in range(10)]

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][tmp_f_lst], mydf.iloc[test_idx][tmp_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = 1, seed = 2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))
    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.abs(np.average(shap_values[0], axis=0))
    shap_imp_cv += shap_values / np.sum(shap_values)


shap_imp_df = pd.DataFrame({'Pro_code': tmp_f_lst,
                            'ShapValues_cv': shap_imp_cv/10})
shap_imp_df.sort_values(by = 'ShapValues_cv', ascending = False, inplace = True)

tg_imp_cv = normal_imp(tg_imp_cv)
tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_cv = normal_imp(tc_imp_cv)
tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = shap_imp_df, right = tg_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df = pd.merge(left = my_imp_df, right = tc_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df['Ensemble'] = (my_imp_df['ShapValues_cv'] + my_imp_df['TotalGain_cv'] + my_imp_df['TotalCover_cv'])/3
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)

my_imp_df.to_csv(outfile, index = False)

print('finished')



