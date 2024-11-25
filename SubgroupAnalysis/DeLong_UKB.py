
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
import time
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/'
outfile = dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB.csv'

mydf = pd.read_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/Pred_UKB.csv')
cols_lst = mydf.columns.tolist()[6:]
nb_preds = len(cols_lst)

fold_id_lst = [i for i in range(10)]

mydf_male = mydf.loc[mydf.DM_GENDER == 1]
mydf_male.reset_index(inplace = True, drop = True)

mydf_female = mydf.loc[mydf.DM_GENDER == 0]
mydf_female.reset_index(inplace = True, drop = True)

mydf_young60 = mydf.loc[mydf.DM_AGE<60]
mydf_young60.reset_index(inplace = True, drop = True)

mydf_old60 = mydf.loc[mydf.DM_AGE>=60]
mydf_old60.reset_index(inplace = True, drop = True)

mydf_young65 = mydf.loc[mydf.DM_AGE<=65]
mydf_young65.reset_index(inplace = True, drop = True)

mydf_old65 = mydf.loc[mydf.DM_AGE>=66]
mydf_old65.reset_index(inplace = True, drop = True)


delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
delong_df.columns = cols_lst
delong_df.index = cols_lst

for i in range(nb_preds):
    for j in range(nb_preds):
        try:
            tmpdf = mydf[['target_y', cols_lst[i], cols_lst[j]]]
            tmpdf.dropna(how='any', inplace=True)
            tmpdf.reset_index(inplace=True, drop=True)
            log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
            delong_df.iloc[i, j] = 10 ** log10_p[0][0]
            print(str(i) + ' ' + str(j))
        except:
            delong_df.iloc[i, j] = np.nan

delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_all.csv', index = True)



delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
delong_df.columns = cols_lst
delong_df.index = cols_lst

for i in range(nb_preds):
    for j in range(nb_preds):
        try:
            tmpdf = mydf_male[['target_y', cols_lst[i], cols_lst[j]]]
            tmpdf.dropna(how='any', inplace=True)
            tmpdf.reset_index(inplace=True, drop=True)
            log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
            delong_df.iloc[i, j] = 10 ** log10_p[0][0]
            print(str(i) + ' ' + str(j))
        except:
            delong_df.iloc[i, j] = np.nan

delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_male.csv', index = True)


delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
delong_df.columns = cols_lst
delong_df.index = cols_lst

for i in range(nb_preds):
    for j in range(nb_preds):
        try:
            tmpdf = mydf_female[['target_y', cols_lst[i], cols_lst[j]]]
            tmpdf.dropna(how='any', inplace=True)
            tmpdf.reset_index(inplace=True, drop=True)
            log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
            delong_df.iloc[i, j] = 10 ** log10_p[0][0]
            print(str(i) + ' ' + str(j))
        except:
            delong_df.iloc[i, j] = np.nan

delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_female.csv', index = True)


delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
delong_df.columns = cols_lst
delong_df.index = cols_lst

for i in range(nb_preds):
    for j in range(nb_preds):
        try:
            tmpdf = mydf_young65[['target_y', cols_lst[i], cols_lst[j]]]
            tmpdf.dropna(how='any', inplace=True)
            tmpdf.reset_index(inplace=True, drop=True)
            log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
            delong_df.iloc[i, j] = 10 ** log10_p[0][0]
            print(str(i) + ' ' + str(j))
        except:
            delong_df.iloc[i, j] = np.nan

delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_young65.csv', index = True)


delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
delong_df.columns = cols_lst
delong_df.index = cols_lst

for i in range(nb_preds):
    for j in range(nb_preds):
        try:
            tmpdf = mydf_old65[['target_y', cols_lst[i], cols_lst[j]]]
            tmpdf.dropna(how='any', inplace=True)
            tmpdf.reset_index(inplace=True, drop=True)
            log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
            delong_df.iloc[i, j] = 10 ** log10_p[0][0]
            print(str(i) + ' ' + str(j))
        except:
            delong_df.iloc[i, j] = np.nan

delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_old65.csv', index = True)

