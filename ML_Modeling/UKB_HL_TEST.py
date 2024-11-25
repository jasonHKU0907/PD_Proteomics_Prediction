

import pandas as pd
import numpy as np
from scipy.stats import chi2

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Revision/Results/'
mydf = pd.read_csv(dpath + 'UKB_ALL/PRO_PANEL/ProPANEL_PredProbs.csv')
#mydf = pd.read_csv(dpath + 'UKB_ALL/FULL/FULL_PredProbs.csv')

pihat = mydf.y_pred_probs
pihatcat=pd.cut(pihat, np.percentile(pihat,[0,25,50,75,100]),labels=False,include_lowest=True) #here I've chosen only 4 groups


meanprobs =[0]*4
expevents =[0]*4
obsevents =[0]*4
meanprobs2=[0]*4
expevents2=[0]*4
obsevents2=[0]*4

for i in range(4):
   meanprobs[i]=np.mean(pihat[pihatcat==i])
   expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
   obsevents[i]=np.sum(mydf.target_y[pihatcat==i])
   meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
   expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
   obsevents2[i]=np.sum(1-mydf.target_y[pihatcat==i])


data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
data2={'expevents':expevents,'expevents2':expevents2}
data3={'obsevents':obsevents,'obsevents2':obsevents2}
m=pd.DataFrame(data1)
e=pd.DataFrame(data2)
o=pd.DataFrame(data3)

tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) #the statistic for the test, which follows,under the null hypothesis, the chi-squared distribution with degrees of freedom equal to amount of groups - 2
pvalue=1-chi2.cdf(tt,2)
pvalue






import pandas as pd
import numpy as np
from scipy.stats import chi2

dpath = '/Volumes/JasonWork/Projects/PD_Proteomics/Results/'
mydf = pd.read_csv(dpath + 'UKB_ALL/PRO_PANEL/ProPANEL_PredProbs.csv')
outfile = dpath + 'UKB_ALL/PRO_PANEL/ProPANEL_SummaryStatistics.csv'
mydf = pd.read_csv(dpath + 'UKB_ALL/FULL/FULL_PredProbs.csv')
outfile = dpath + 'UKB_ALL/FULL/FULL_SummaryStatistics.csv'
fold_id_lst = [i for i in range(10)]

pihat = mydf.y_pred_probs
pihatcat=pd.cut(pihat, np.percentile(pihat,[0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),labels=False,include_lowest=True) #here I've chosen only 4 groups


meanprobs =[0]*9
expevents =[0]*9
obsevents =[0]*9
meanprobs2=[0]*9
expevents2=[0]*9
obsevents2=[0]*9

for i in range(9):
   meanprobs[i]=np.mean(pihat[pihatcat==i])
   expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
   obsevents[i]=np.sum(mydf.target_y[pihatcat==i])
   meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
   expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
   obsevents2[i]=np.sum(1-mydf.target_y[pihatcat==i])


data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
data2={'expevents':expevents,'expevents2':expevents2}
data3={'obsevents':obsevents,'obsevents2':obsevents2}
m=pd.DataFrame(data1)
e=pd.DataFrame(data2)
o=pd.DataFrame(data3)

tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) #the statistic for the test, which follows,under the null hypothesis, the chi-squared distribution with degrees of freedom equal to amount of groups - 2
pvalue=1-chi2.cdf(tt,2)
pvalue