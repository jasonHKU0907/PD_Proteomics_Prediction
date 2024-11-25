

import glob
import os
import numpy as np
import pandas as pd
import re
import math
dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
ppmi_df = pd.read_csv(dpath + 'PPMI/Raw/bl_Plasma_protein_data.csv')
ukb_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')

tmp_df = ppmi_df.copy()
pro_f_lst = ukb_df.columns.tolist()[:-1]
pro_f_lst = [f for f in pro_f_lst if f in ppmi_df.columns.tolist()]
out_df = ppmi_df[['PATNO']]

#pro_f_lst = ['NEFL', 'HPGDS', 'IL13RA1', 'PTPRN2', 'TPPP3', 'TNXB', 'RAB6A']

for pro_f in pro_f_lst:
    ukb_df[pro_f] = ukb_df[pro_f] / np.abs(ukb_df[pro_f]) * np.log(np.abs(ukb_df[pro_f]) + 1)
    ubd = ukb_df[pro_f].mean() + ukb_df[pro_f].std() * 4
    lbd = ukb_df[pro_f].mean() - ukb_df[pro_f].std() * 4
    tmp_df[pro_f] = tmp_df[pro_f] / np.abs(tmp_df[pro_f]) * np.log(np.abs(tmp_df[pro_f]) + 1)
    tmp_df[pro_f].iloc[tmp_df[pro_f] > ubd] = np.nan
    tmp_df[pro_f].iloc[tmp_df[pro_f] < lbd] = np.nan
    tmp_f = np.round((tmp_df[pro_f] - np.mean(ukb_df[pro_f])) / ukb_df[pro_f].std(), 5)
    out_df[pro_f] = np.array(tmp_f)

out_df.to_csv(dpath + 'PPMI/PPMI_ProteomicsData.csv', index = False)

