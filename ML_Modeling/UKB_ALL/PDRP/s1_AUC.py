

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision/Revision_R2/'
outfile = dpath + 'Results/UKB_10YEARS/FULL/s1_AUC.csv'

pro_f_df = pd.read_csv(dpath + 'Results/UKB_10YEARS/PRO_PANEL/s5_AccAUC_TotalGain.csv')
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

target_df = pd.read_csv(dpath + 'Data/TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Data/Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)
pro_df = pd.read_csv(dpath + 'Data/Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'Data/Proteomics/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])

mydf = pd.merge(target_df, cov_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, pro_df, how = 'inner', on = ['eid'])

fold_id_lst = [i for i in range(10)]

auc_mean_lst, auc_std_lst = [], []

for pro_f in tqdm(pro_f_lst+cov_f_lst):
    tmpdf = mydf[['target_y', 'Region_code', pro_f]]
    tmpdf.rename(columns={pro_f: "target_pro"}, inplace=True)
    rm_idx = tmpdf.index[tmpdf.target_pro.isnull() == True]
    tmpdf = tmpdf.drop(rm_idx, axis=0)
    tmpdf.reset_index(inplace=True, drop = True)
    tmp_auc_lst = []
    tmp_auc = roc_auc_score(tmpdf.target_y, tmpdf.target_pro)
    if tmp_auc>=0.5:
        for fold_id in fold_id_lst:
            test_idx = tmpdf['Region_code'].index[tmpdf['Region_code'] == fold_id]
            y_pred, y_test = tmpdf.iloc[test_idx].target_pro, tmpdf.iloc[test_idx].target_y
            tmp_auc_lst.append(roc_auc_score(y_test, y_pred))
    else:
        tmpdf['target_pro'] = -tmpdf.target_pro
        for fold_id in fold_id_lst:
            test_idx = tmpdf['Region_code'].index[tmpdf['Region_code'] == fold_id]
            y_pred, y_test = tmpdf.iloc[test_idx].target_pro, tmpdf.iloc[test_idx].target_y
            tmp_auc_lst.append(roc_auc_score(y_test, y_pred))
    auc_mean_lst.append(np.mean(tmp_auc_lst))
    auc_std_lst.append(np.std(tmp_auc_lst))

myout_df = pd.DataFrame({'AUC_mean':auc_mean_lst, 'AUC_std':auc_std_lst})
myout_df['Pro_code'] = pro_f_lst+cov_f_lst

myout_df = pd.merge(myout_df, pro_dict, how = 'left', on = ['Pro_code'])

myout_df = myout_df[['Pro_code', 'AUC_mean', 'AUC_std', 'Pro_definition']]
myout_df.to_csv(outfile, index = False)


