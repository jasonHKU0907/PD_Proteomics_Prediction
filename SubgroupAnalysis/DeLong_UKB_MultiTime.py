
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


for yr in [2, 4, 6, 8, 10, 12, 14, 100]:
    tmpdf0 = mydf.copy()
    tmpdf0['target_y'].loc[tmpdf0.BL2Target_yrs >= yr] = 0
    delong_df = pd.DataFrame(np.zeros((nb_preds, nb_preds)))
    delong_df.columns = cols_lst
    delong_df.index = cols_lst
    for i in range(nb_preds):
        for j in range(nb_preds):
            try:
                tmpdf = tmpdf0[['target_y', cols_lst[i], cols_lst[j]]]
                tmpdf.dropna(how='any', inplace=True)
                tmpdf.reset_index(inplace=True, drop=True)
                log10_p = delong_roc_test(tmpdf.target_y, tmpdf.iloc[:, 1], tmpdf.iloc[:, 2])
                delong_df.iloc[i, j] = 10 ** log10_p[0][0]
                print(str(i) + ' ' + str(j))
            except:
                delong_df.iloc[i, j] = np.nan
    delong_df.to_csv(dpath + 'Neurology_Revision/Results/SubGroupAnalysis/DeLong_UKB_MultiTime_'+str(yr)+'yrs.csv', index=True)


