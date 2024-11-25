
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utility.Training_Utilities import *
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

def best_analysts(my_X, f_lst, df_auc):
    f_df = pd.DataFrame({'Pro_code':  my_X.columns[f_lst]})
    merged_df = pd.merge(f_df, df_auc, how='inner', on=['Pro_code'])
    merged_df.sort_values(by = 'AUC_mean', ascending=False)
    return merged_df.Pro_code[0]

def get_imp_analy(Imp_df, top_prop):
    imp_score, iter = 0, 0
    while imp_score < top_prop:
        imp_score += Imp_df.TotalGain_cv[iter]
        iter+=1
    return iter+1

top_prop=0.5


dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Neurology_Revision/Results/ProPanelSelection/XGB_ProPANEL/'

outfile = outpath + 's3_rmMultiColinearity.csv'
outimg = outpath + 's3_rmMultiColinearity.png'
Imp_df = pd.read_csv(outpath + 's2_PROImportance.csv')
top_nb = get_imp_analy(Imp_df, top_prop)
Imp_df = Imp_df.iloc[:top_nb,:]
pro_f_lst = Imp_df.Pro_code.tolist()

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv', usecols=['eid'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
target_df = pd.read_csv(dpath + 'TargetOutcomes/PD/PD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])

my_X = mydf[pro_f_lst]
y = mydf.target_y
my_label = Imp_df.Pro_code.tolist()

corr = np.array(my_X.corr(method='spearman'))
#corr = np.array(X.corr(method='pearson'))
corr = np.nan_to_num(corr)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
dendro = hierarchy.dendrogram(dist_linkage, labels=pro_f_lst, ax=ax2)
ax2.set_xticklabels(dendro["ivl"], rotation=60, fontsize=5, horizontalalignment='right')
ax2.axhline(y = 0.6, color = 'r', linewidth=2, linestyle = '--')
dendro_idx = np.arange(0, len(dendro["ivl"]))
ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax1.set_xticks(dendro_idx)
ax1.set_yticks(dendro_idx)
ax1.set_xticklabels(dendro["ivl"], rotation=60, fontsize=5, horizontalalignment='right')
ax1.set_yticklabels(dendro["ivl"], fontsize=5)
fig.tight_layout()
plt.show()
plt.savefig(outimg)

cluster_ids = hierarchy.fcluster(dist_linkage, .6, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

Imp_df['Cluster_ids'] = cluster_ids
selected_f = [best_analysts(my_X, v, Imp_df) for v in cluster_id_to_feature_ids.values()]
select = ['*' if item in selected_f else '' for item in Imp_df.Pro_code]
Imp_df['Selected'] = select

myout_df = Imp_df[['Pro_code', 'TotalGain_cv', 'AUC_mean', 'AUC_std', 'Cluster_ids', 'Selected', 'Pro_definition']]
myout_df.to_csv(outfile, index = False)

print('Finished')
