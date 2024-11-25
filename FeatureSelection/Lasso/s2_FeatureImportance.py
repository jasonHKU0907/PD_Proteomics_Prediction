
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from xgboost import XGBClassifier
import warnings
import re
import shap
from sklearn.linear_model import LogisticRegression
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Neurology_Revision/Results/ProPanelSelection/LASSO/'
outfile = outpath + 'LASSO_Selection.csv'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_f_lst = pro_df.columns.tolist()[1:]
for pro_f in pro_f_lst:
    pro_df[pro_f].fillna(pro_df[pro_f].mean(), inplace=True)

target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])
auc_df = pd.read_csv(outpath + 's1_Protein_AUC.csv')

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])

log_clf = LogisticRegression(C = 0.1, penalty= 'l1', solver= 'liblinear')
log_clf.fit(mydf[pro_f_lst], mydf.target_y)
my_imp_df = pd.DataFrame({'Pro_code':pro_f_lst, 'Pro_imp':log_clf.coef_.tolist()[0]})
my_imp_df = pd.merge(my_imp_df, auc_df, how = 'left', on=['Pro_code'])

my_imp_df.to_csv(outfile)



