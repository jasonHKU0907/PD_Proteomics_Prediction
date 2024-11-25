

import glob
import os
import numpy as np
import pandas as pd
import re
import math
dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
ppmi_df = pd.read_csv(dpath + 'PPMI/Raw/ppmi_cov_data.csv')

out_df = ppmi_df[['PATNO']]
out_df['PD_ProHea'] = ppmi_df['group'].copy()
out_df['PD_ProHea'].replace([1, 4, 2], [1, 0, 0], inplace = True)

out_df['PDPro_Hea'] = ppmi_df['group'].copy()
out_df['PDPro_Hea'].replace([1, 4, 2], [1, 1, 0], inplace = True)

out_df['PD_Hea'] = ppmi_df['group'].copy()
out_df['PD_Hea'].replace([1, 4, 2], [1, 100, 0], inplace = True)

out_df['PD_Pro'] = ppmi_df['group'].copy()
out_df['PD_Pro'].replace([1, 4, 2], [1, 0, 100], inplace = True)

out_df['Pro_Hea'] = ppmi_df['group'].copy()
out_df['Pro_Hea'].replace([1, 4, 2], [100, 1, 0], inplace = True)

out_df.to_csv(dpath + 'PPMI/PPMI_PD_outcomes.csv', index = False)

