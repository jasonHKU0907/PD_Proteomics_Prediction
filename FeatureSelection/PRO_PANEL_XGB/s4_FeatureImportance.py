

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from xgboost import XGBClassifier
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Neurology_Revision/Results/ProPanelSelection/XGB_ProPANEL/'
outfile = outpath + 's4_PROImportance.csv'

pro_df = pd.read_csv(outpath + 's3_rmMultiColinearity.csv', usecols=['Pro_code', 'Selected'])
pro_f_lst = pro_df.loc[pro_df.Selected == '*'].Pro_code.tolist()
auc_df = pd.read_csv(outpath + 's1_Protein_AUC.csv')

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv', usecols=['eid'] + pro_f_lst)
target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'min_child_weight': 10,
             'subsample': 0.7,
             'eta': 0.01}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()
fold_id_lst = [i for i in range(10)]

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[test_idx][pro_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_xgb = XGBClassifier(objective='binary:logistic', nthread=8, eval_metric='auc', verbosity=1, seed=2023)
    my_xgb.set_params(**my_params)
    my_xgb.fit(X_train, y_train)
    totalgain_imp = my_xgb.get_booster().get_score(importance_type='total_gain')
    totalcover_imp = my_xgb.get_booster().get_score(importance_type='cover')
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))

tg_imp_cv = normal_imp(tg_imp_cv)
tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_cv = normal_imp(tc_imp_cv)
tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = tg_imp_df, right = tc_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)
my_imp_df = pd.merge(my_imp_df, auc_df, how = 'left', on=['Pro_code'])

my_imp_df.to_csv(outfile, index = False)

print('finished')



