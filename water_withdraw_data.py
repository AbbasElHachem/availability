
# coding: utf-8

# In[52]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[157]:

df_data=pd.read_csv(r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\agriculture_industry_municipal_water_demand_original_sorted.csv', index_col=0,
                   usecols=[0, 1, 2,3])


# In[159]:

df_data.describe()


# In[146]:

df_data0 = df_data
df_data_abv_thr=df_data[df_data.iloc[:,0] > 60]


# In[147]:

x_vals = df_data_abv_thr.index
df_data_abv_thr.shape[0] / df_data0.shape[0]


# In[148]:

val_agriculture = df_data_abv_thr.iloc[:,0]
ind_vals = df_data_abv_thr.iloc[:,1]
munic_vls = df_data_abv_thr.iloc[:,2]
width = 0.5 

np.mean(val_agriculture), np.mean(ind_vals), np.mean(munic_vls)


# In[149]:

plt.figure(figsize=(60,40), dpi=600)
#p1 = plt.bar(x_vals, val_agriculture, 0.5, color='darkred', alpha=0.75)
#p2 = plt.bar(x_vals, ind_vals, 0.5, bottom=val_agriculture, color='darkblue',  alpha=0.75)
#p3 = plt.bar(x_vals, munic_vls, .5, bottom=ind_vals+val_agriculture, color='darkgreen',  alpha=0.75)
xticks = np.arange(0, 101, 10)
p1 = plt.barh(x_vals, val_agriculture, 0.5, color='darkred')
p2 = plt.barh(x_vals, ind_vals, 0.5, left=val_agriculture, color='darkblue')
p3 = plt.barh(x_vals, munic_vls, .5, left=ind_vals+val_agriculture, color='darkgreen')

plt.xlabel(' Water withdrawal as % of total water withdrawal (%)', fontweight='bold', fontsize=14)
plt.title('Water withdrawal per sector', fontweight='bold', fontsize=14)
#plt.xticks(x_vals, rotation=65, fontweight='bold', fontsize=12)
plt.yticks(fontweight='bold',fontsize=14)
plt.xlim([0, 100])
plt.grid(alpha=0.5)
#plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Agriculture', 'Industry', 'Municipal'), fontsize=12)

plt.savefig(r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\agriculture_industry_municipal_water_demand.png',
           frameon=True, papertype='a4',
                       bbox_inches='tight', pad_inches=.2)
plt.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



