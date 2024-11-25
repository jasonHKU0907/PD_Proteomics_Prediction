

import glob
import os
import numpy as np
import pandas as pd
import re
import math
dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
ppmi_df = pd.read_csv(dpath + 'PPMI/Raw/ppmi_cov_data.csv')
ppmi_bf_df = pd.read_csv(dpath + 'PPMI/Raw/ppmi_bl_Serum_data.csv')
ppmi_bf_df.rename(columns = {'SerumIGF-1': 'BF_EC_IGF1'}, inplace = True)

out_df = ppmi_df[['PATNO']]
out_df['DM_AGE'] = ppmi_df['AGE_AT_VISIT']
out_df['DM_GENDER'] = ppmi_df['SEX']
out_df['DM_EDUC'] = ppmi_df['EDUCYRS'] - 10
out_df['DM_EDUC'].loc[out_df['DM_EDUC']<=0] = 0
out_df = pd.merge(out_df, ppmi_bf_df[['PATNO', 'BF_EC_IGF1']], how = 'left', on = ['PATNO'])
out_df['MH_NBMED'] = 1
out_df.to_csv(dpath + 'PPMI/PPMI_PANEL.csv', index = False)

