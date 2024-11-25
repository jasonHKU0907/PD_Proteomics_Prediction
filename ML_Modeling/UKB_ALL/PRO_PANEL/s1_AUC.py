

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
pd.options.mode.chained_assignment = None  # default='warn'


dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_f_lst = pro_df.columns[1:].tolist()
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])

target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['eid'])
fold_id_lst = [i for i in range(10)]

auc_mean_lst, auc_std_lst = [], []

for pro_f in tqdm(pro_f_lst):
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
myout_df['Pro_code'] = pro_f_lst

myout_df = pd.merge(myout_df, pro_dict, how = 'left', on = ['Pro_code'])

myout_df = myout_df[['Pro_code', 'AUC_mean', 'AUC_std', 'Pro_definition']]
myout_df.to_csv(outpath + 'UKB_ALL/PRO_PANEL/s1_Protein_AUC.csv', index = False)


