
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import operator

pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outpath = dpath + 'Neurology_Revision/Results/ProPanelSelection/XGB_PDRP/'
outfile = outpath + 's4_PDRP_PredProbs_XGB.csv'

imp_f_df = pd.read_csv(outpath + 's3_AccAUC_TotalGain.csv')
nb_imp_f = get_top_pros(imp_f_df)
my_f_lst = imp_f_df.Pro_code.tolist()[:nb_imp_f]
pro_f_lst = [f for f in my_f_lst if '_' not in f]
cov_f_lst = [f for f in my_f_lst if f not in pro_f_lst]

target_df = pd.read_csv(dpath + 'Revision_R2/Data/TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
pro_df = pd.read_csv(dpath + 'Revision_R2/Data/Proteomics/ProteomicsData.csv', usecols = ['eid']+pro_f_lst)
cov_df = pd.read_csv(dpath + 'Revision_R2/Data/Covariates/PANEL_raw.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])

'''
nb_models = 100
mykf = StratifiedKFold(n_splits = 10, random_state = 2022, shuffle = True)

params_dict = {'n_estimators': np.linspace(100, 500, 9).astype('int32').tolist(),
               'max_depth': np.linspace(6, 15, 6).astype('int32').tolist(),
               'min_child_weight': np.linspace(1, 15, 6).astype('int32').tolist(),
               'subsample': np.round(np.linspace(0.6, 1.0, 5), 2).tolist(),
               'eta': [0.1, 0.05, 0.01, 0.005]}

selected_params = select_params_combo(params_dict, 100)
AUC_list = []
X = mydf[my_f_lst]
y = mydf.target_y

for my_params in selected_params:
    AUC_cv = []
    for train_idx, test_idx in mykf.split(X, y):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 8, eval_metric = 'auc',
                               verbosity = 1, seed = 2020)
        my_xgb.set_params(**my_params)
        my_xgb.fit(X_train, y_train)
        y_pred_prob = my_xgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
        print(roc_auc_score(y_test, y_pred_prob))
    AUC_list.append(np.round(np.average(AUC_cv), 4))

index, best_auc = max(enumerate(AUC_list), key = operator.itemgetter(1))
best_params = selected_params[index]
'''

best_params = {'n_estimators': 500, 'max_depth': 15, 'min_child_weight': 15,
               'subsample': 0.7, 'eta': 0.01}


fold_id_lst = [i for i in range(10)]
eid_lst, region_lst = [], []
y_test_lst, y_pred_lst = [], []

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][my_f_lst], mydf.iloc[test_idx][my_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_xgb = XGBClassifier(objective='binary:logistic', nthread=8, eval_metric='auc', verbosity=1, seed=2023)
    my_xgb.set_params(**best_params)
    my_xgb.fit(X_train, y_train)
    y_pred_prob = my_xgb.predict_proba(X_test)[:, 1].tolist()
    y_pred_lst += y_pred_prob
    y_test_lst += mydf.target_y.iloc[test_idx].tolist()
    eid_lst += mydf.eid.iloc[test_idx].tolist()
    region_lst += mydf.Region_code.iloc[test_idx].tolist()

myout_df = pd.DataFrame([eid_lst, region_lst, y_test_lst, y_pred_lst]).T
myout_df.columns = ['eid', 'Region_code', 'target_y', 'y_pred_probs']
myout_df[['eid', 'Region_code']] = myout_df[['eid', 'Region_code']].astype('int')

myout_df.to_csv(outfile, index = False)

roc_auc_score(myout_df.target_y, myout_df.y_pred_prob)
