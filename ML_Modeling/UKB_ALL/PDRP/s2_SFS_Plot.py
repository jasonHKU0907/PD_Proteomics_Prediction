

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision_R2/Results/'
output_img_dir = dpath + 'UKB_ALL/FULL/s2_AccAUC_TotalGain.png'

pro_imp_df = pd.read_csv(dpath + 'UKB_ALL/FULL/s1_FeaImportance.csv')
pro_imp_df.rename(columns = {'TotalGain_cv': 'Pro_imp'}, inplace = True)
pro_auc_df = pd.read_csv(dpath + 'UKB_ALL/FULL/s2_AccAUC_TotalGain.csv')
f_dict = pd.read_csv(dpath + 'Plots/CovariateInfo.csv')
mydf = pd.merge(pro_auc_df, pro_imp_df, how = 'left', on = ['Pro_code'])
mydf = pd.merge(mydf, f_dict, how = 'left', left_on = ['Pro_code'], right_on = ['Covariate_code'])
mydf['Predictor'][mydf.Predictor.isnull()] = mydf['Pro_code'][mydf.Predictor.isnull()]
mydf['AUC_lower'] = mydf['AUC_mean'] - 1.96*mydf['AUC_std']
mydf['AUC_upper'] = mydf['AUC_mean'] + 1.96*mydf['AUC_std']
mydf['pro_idx'] = [i for i in range(1, len(mydf)+1)]
nb_f = get_top_pros(mydf)
mydf = mydf.iloc[:35, :]

mydf['Predictor'].replace(['EDA2R', 'NEFL', 'IL13RA1', 'BAG3', 'SCARF2', 'ITGAV', 'MERTK',
                           'HPGDS', 'SCG2', 'CXCL9', 'HNMT', 'TNXB', 'KLK8', 'NCAM1', 'CDH15',
                           'ACE2', 'TNFSF13', 'LXN', 'XG', 'TPPP3', 'NPPB', 'WARS', 'Creatinine'],
                          ['Plasma EDA2R', 'Plasma NEFL', 'Plasma IL13RA1', 'Plasma BAG3',
                           'Plasma SCARF2', 'Plasma ITGAV', 'Plasma MERTK', 'Plasma HPGDS',
                           'Plasma SCG2', 'Plasma CXCL9', 'Plasma HNMT', 'Plasma TNXB',
                           'Plasma KLK8', 'Plasma NCAM1', 'Plasma CDH15', 'Plasma ACE2',
                           'Plasma TNFSF13', 'Plasma LXN', 'Plasma XG', 'Plasma TPPP3',
                           'Plasma NPPB', 'Plasma WARS', 'Serum creatinine'], inplace = True)


fig, ax = plt.subplots(figsize = (18, 6.5))
palette = sns.color_palette("Blues",n_colors=len(mydf))
palette.reverse()
sns.barplot(ax=ax, x = "Predictor", y = "Pro_imp", palette=palette, data=mydf.sort_values(by="Pro_imp", ascending=False))
y_imp_up_lim = round(mydf['Pro_imp'].max() + 0.01, 2)
ax.set_ylim([0, y_imp_up_lim])
ax.tick_params(axis='y', labelsize=14)
ax.set_xticklabels(mydf['Predictor'], rotation=45, fontsize=14, horizontalalignment='right')
my_col = ['r']*nb_f + ['k']*(len(mydf)-nb_f)
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_col):
    ticklabel.set_color(tickcolor)

ax.set_ylabel('Predictor Importance', weight='bold', fontsize=18)
ax.grid(which='minor', alpha=0.2, linestyle=':')
ax.grid(which='major', alpha=0.5,  linestyle='--')
ax.set_xlabel('')
ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.plot(np.arange(nb_f+1), mydf['AUC_mean'][:nb_f+1], 'red', alpha = 0.8, marker='o')
ax2.plot(np.arange(nb_f+1, len(mydf)), mydf['AUC_mean'][nb_f+1:], 'black', alpha = 0.8, marker='o')
ax2.plot([nb_f, nb_f+1], mydf['AUC_mean'][nb_f:nb_f+2], 'black', alpha = 0.8, marker='o')
plt.fill_between(mydf['pro_idx']-1, mydf['AUC_lower'], mydf['AUC_upper'], color = 'tomato', alpha = 0.2)
ax2.set_ylabel('Cumulative AUC', weight='bold', fontsize=18)
ax2.set_ylim([0.65, 0.91])
ax2.get_yaxis().set_ticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
ax2.tick_params(axis='y', labelsize=14)
ax2.grid(which='minor', alpha=0.2, linestyle=':')
ax2.grid(which='major', alpha=0.5,  linestyle='--')

fig.tight_layout()
plt.xlim([-.6, len(mydf)-.2])
plt.savefig(output_img_dir, dpi=400)
