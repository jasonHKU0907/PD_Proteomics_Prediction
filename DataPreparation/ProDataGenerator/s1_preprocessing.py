

import glob
import os
import numpy as np
import pandas as pd
import re
import math

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Proteomics/'
mydf = pd.read_csv(dpath + 'Raw/ProteomicsData.csv')
pro_f_lst = mydf.columns.tolist()[:-1]

mydf_out = mydf['eid'].copy()

for col in pro_f_lst:
    tmp_col = mydf[col]/np.abs(mydf[col])*np.log(np.abs(mydf[col]) + 1)
    ubd = tmp_col.mean() + tmp_col.std() * 4
    lbd = tmp_col.mean() - tmp_col.std() * 4
    tmp_col.iloc[tmp_col > ubd] = np.nan
    tmp_col.iloc[tmp_col < lbd] = np.nan
    tmp_col = np.round((tmp_col - np.mean(tmp_col)) / tmp_col.std(), 5)
    mydf_out = pd.concat([mydf_out, tmp_col], axis = 1)

mydf_out.to_csv(dpath + 'ProteomicsData.csv', index = False)

