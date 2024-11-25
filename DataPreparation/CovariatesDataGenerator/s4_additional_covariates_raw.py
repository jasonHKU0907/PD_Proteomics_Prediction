

import glob
import os
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer

def get_standardization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        ubd = tmp_df[col].quantile(0.995)
        lbd = tmp_df[col].quantile(0.005)
        tmp_df[col].iloc[tmp_df[col]>ubd] = ubd
        tmp_df[col].iloc[tmp_df[col]<lbd] = lbd
        tmp_df[col] = (tmp_df[col] - lbd) / (ubd - lbd)
    return tmp_df


def read_data(FieldID_lst, feature_df, eid_df):
    subset_df = feature_df[feature_df['Field_ID'].isin(FieldID_lst)]
    subset_dict = {k: ['eid'] + g['Field_ID'].tolist() for k, g in subset_df.groupby('Subset_ID')}
    subset_lst = list(subset_dict.keys())
    my_df = eid_df
    for subset_id in subset_lst:
        tmp_dir = dpath + 'UKB_subset_' + str(subset_id) + '.csv'
        tmp_f = subset_dict[subset_id]
        tmp_df = pd.read_csv(tmp_dir, usecols=tmp_f)
        my_df = pd.merge(my_df, tmp_df, how='inner', on=['eid'])
    return my_df

def get_days_intervel(start_date_var, end_date_var, df):
    start_date = pd.to_datetime(df[start_date_var], dayfirst=True)
    end_date = pd.to_datetime(df[end_date_var], dayfirst=True)
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    my_yrs = [ele/365 for ele in days]
    return pd.DataFrame(my_yrs)

def get_binary(var_source, df):
    tmp_binary = df[var_source].copy()
    tmp_binary.loc[tmp_binary >= -1] = 1
    tmp_binary.replace(np.nan, 0, inplace=True)
    return tmp_binary

dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')

f_lst = ['924-0.0', '46-0.0', '47-0.0', '95-0.0', '102-0.0', '22034-0.0', '1170-0.0', '1190-0.0', '26260-0.0',
         '3062-0.0', '3063-0.0', '20258-0.0', '30850-0.0', '23115-0.0', '23111-0.0', '30260-0.0']
mydf = read_data(f_lst, feature_df, eid_df)

mydf['924-0.0'].replace([-3, -7], np.nan, inplace = True)
mydf['PM_HGS'] = (mydf['46-0.0'] + mydf['47-0.0'])/2
mydf['1170-0.0'].replace([-1, -3], np.nan, inplace = True)
mydf['1190-0.0'].replace(-3, np.nan, inplace = True)

pr_idx = mydf.index[mydf['102-0.0'].isnull()]
mydf.loc[pr_idx, '102-0.0'] = mydf.loc[pr_idx, '95-0.0']

mydf['PM_LFP'] = (mydf['23115-0.0'] + mydf['23111-0.0'])/2

myoutdf = mydf[['eid', '924-0.0', 'PM_HGS', '102-0.0', '22034-0.0', '1170-0.0', '1190-0.0', '26260-0.0',
                '3062-0.0', '3063-0.0', '20258-0.0', '30850-0.0', '30260-0.0', 'PM_LFP']]
myoutdf.rename(columns = {'924-0.0': 'LS_WP', '102-0.0': 'PM_PR', '22034-0.0': 'LS_AT',
                          '1170-0.0': 'LS_MG', '1190-0.0': 'LS_DN', '26260-0.0':'PRS',
                          '3062-0.0': 'PM_FVC', '3063-0.0': 'PM_FEV1', '20258-0.0': 'PM_FEV1_FVC',
                          '30850-0.0': 'BF_EC_TE', '30260-0.0': 'BF_BC_MRV'}, inplace = True)

myoutdf.to_csv('/Volumes/JasonWork/Projects/PD_Proteomics/Data/Covariates/RAW/AdditionalCov.csv', index = False)
