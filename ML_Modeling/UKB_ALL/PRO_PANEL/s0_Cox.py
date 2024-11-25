

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_f_lst = pro_df.columns[1:].tolist()
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])

cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv')
m_f_lst = ['age', 'sex', 'ethn', 'educ', 'tdi', 'smk', 'alc', 'reg_pa', 'sbp', 'bmi', 'Region_code']

target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
target_df.BL2Target_yrs.describe()

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])

myout_df, pro_out_lst = pd.DataFrame(), []

for pro_f in tqdm(pro_f_lst):
    tmpdf_f = ['target_y', 'BL2Target_yrs', pro_f] + m_f_lst
    tmpdf = mydf[tmpdf_f]
    tmpdf.rename(columns={pro_f: "target_pro"}, inplace=True)
    rm_idx = tmpdf.index[tmpdf.target_pro.isnull() == True]
    tmpdf = tmpdf.drop(rm_idx, axis=0)
    tmpdf.reset_index(inplace=True)
    cph = CoxPHFitter()
    my_formula = "age + sex + C(ethn) + educ + tdi + smk + alc + bmi + Region_code + target_pro"
    try:
        cph.fit(tmpdf, duration_col = 'BL2Target_yrs', event_col = 'target_y', formula=my_formula)
        hr = cph.hazard_ratios_.target_pro
        lbd = np.exp(cph.confidence_intervals_).iloc[10, 0]
        ubd = np.exp(cph.confidence_intervals_).iloc[10, 1]
        pval = cph.summary.p.target_pro
        myout = pd.DataFrame([hr, lbd, ubd, pval])
        myout_df = pd.concat((myout_df, myout.T), axis=0)
        pro_out_lst.append(pro_f)
    except:
        print(pro_f)

myout_df.columns = ['HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val']
myout_df['Pro_code'] = pro_out_lst
_, p_f_fdr = fdrcorrection(myout_df.HR_p_val.fillna(1))
_, p_f_bfi = bonferroni_correction(myout_df.HR_p_val.fillna(1), alpha=0.01)

myout_df['p_val_fdr'] = p_f_fdr
myout_df['p_val_bfi'] = p_f_bfi

myout_df = pd.merge(myout_df, pro_dict, how = 'left', on = ['Pro_code'])

myout_df = myout_df[['Pro_code', 'Pro_definition', 'HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val', 'p_val_fdr', 'p_val_bfi']]
myout_df.to_csv(outpath + 'UKB_ALL/PRO_PANEL/Cox11.csv', index = False)


mydf = pd.read_csv(outpath + 'UKB_ALL/PRO_PANEL/s2_PROImportance.csv')
mydf = pd.merge(mydf, myout_df, how = 'left', on = ['Pro_code'])
mydf['HR'] = np.round(mydf['HR'], 3)
mydf['HR_Lower_CI'] = np.round(mydf['HR_Lower_CI'], 3)
mydf['HR_Upper_CI'] = np.round(mydf['HR_Upper_CI'], 3)
mydf['HR_out'] = [str(mydf.HR.iloc[i]) +' [' + str(mydf.HR_Lower_CI.iloc[i]) + '-' + str(mydf.HR_Upper_CI.iloc[i]) + ']' for i in range(len(mydf))]

mydf.to_csv(outpath + 'UKB_ALL/PRO_PANEL/Cox11111.csv', index = False)