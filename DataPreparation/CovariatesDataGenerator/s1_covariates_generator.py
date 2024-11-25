

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/Covariates/Raw/'

demo_df = pd.read_csv(dpath + 'DemographicInfo.csv')
ls_df = pd.read_csv(dpath + 'LifeStyle.csv', usecols = ['eid', 'SMK_Status', 'ALC_Status'])
pm_df = pd.read_csv(dpath + 'PhysicalMeasurements.csv', usecols = ['eid', 'SBP', 'BMI'])
pa_df = pd.read_csv(dpath + 'Lifestyle_PA.csv', usecols = ['eid', 'RegularPA'])

my_cov_df = pd.merge(demo_df, ls_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, pm_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, pa_df, how = 'left', on = ['eid'])

my_cov_df.rename(columns = {'Age': 'age', 'Gender': 'sex', 'Ethnicity': 'ethn', 'Education':'educ', 'TDI': 'tdi',
                            'SMK_Status': 'smk', 'ALC_Status': 'alc', 'SBP': 'sbp', 'BMI':'bmi', 'RegularPA':'reg_pa'}, inplace = True)

my_cov_df_out = my_cov_df[['eid', 'age', 'sex', 'ethn', 'educ', 'tdi', 'smk', 'alc', 'reg_pa', 'sbp', 'bmi']]

my_cov_df_out.to_csv(outpath + 'Covariates_full_population.csv', index = False)



