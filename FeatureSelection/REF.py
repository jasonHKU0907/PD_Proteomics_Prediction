
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import shap
from sklearn.linear_model import LogisticRegression
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
dpath1 = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'
outfile = '/Volumes/JasonWork/Projects/PD_Proteomics/Neurology_Revision/Results/ProPanelSelection/RFE/FeatureSelection.csv'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_f_lst = pro_df.columns.tolist()[1:]

for pro_f in pro_f_lst:
    pro_df[pro_f].fillna(pro_df[pro_f].mean(), inplace=True)

auc_df = pd.read_csv(dpath1 + 'UKB_ALL/PRO_PANEL/s1_Protein_AUC.csv')

target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])

log_clf = LogisticRegression(C = 0.1, penalty= 'l1', solver= 'liblinear')

selector = RFE(log_clf, step=1)
selector = selector.fit(mydf[pro_f_lst], mydf.target_y)
selector.support_
selector.ranking_
myout = pd.DataFrame({'Pro_code':pro_f_lst, 'Pro_imp':selector.ranking_.tolist()})
myout.to_csv(outfile)

