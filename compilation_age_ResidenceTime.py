# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:52:21 2020

@author: Adrien

Plot of bar chart of velocity / distance / residence time
Compilation of the information from AgragationDataTable_v4 
Data from my model (model + Darcy's law) + Clark + MacFarlane + Thornton + Whittemore + Hegelsen
Best to visualize / compare
Distance along axis/cross section from map "MapProspectus" (cross_section_line), with recharge 
along the outcrop of Dakota aquifer in SW Colorado and discharge around Republican river

 
"""

import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tasgrid as tg
import scipy
#import utm
import peakutils
from peakutils.plot import plot as pplot
import math
import statsmodels.formula.api as sm
from matplotlib import dates as mdates

from datetime import datetime

import mpld3

from functools import reduce
import matplotlib.colors as mcolors
import matplotlib


#%%

plt.close('all')

os.chdir("C:/Users/Reika/Dropbox/RessourceWaterMidcontinent")
#C:/Users/Reika/Dropbox/RessourceWaterMidcontinent/quality_dakota

width = 0.2

data = pd.read_csv("AgragationDataTable_v5.csv")
data=data.set_index('WELL NAME')
data=data.sort_values(by=['DistanceRecharge_km'])

N=data.shape[0]
ind = np.arange(data.shape[0])

#clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
#         (0.7, "green"), (0.75, "blue"), (1, "blue")]
#rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
#
#x = np.arange(N).astype(float)


norm = matplotlib.colors.LogNorm(vmin=1000, vmax=1.5e5, clip=True)
#norm = matplotlib.colors.LogNorm(vmin=min(dataClark['14C age in years from Clark et al. (1998)'].values), vmax=1.5e8, clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
mapper.set_array(data['Residence time in years from my estimation using Darcy Law'].values)


#%% Different plot for each source (model/clark...)

dataNeil=data.dropna(subset=['81Kr age in years from Neil'])
dataNeil=dataNeil[['81Kr age in years from Neil','DistanceRecharge_km']]
dataNeil['some_value_color'] = dataNeil['81Kr age in years from Neil'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
NeilIndex=np.where(~np.isnan(data['81Kr age in years from Neil']))

#
dataClark=data.dropna(subset=['14C age in years from Clark et al. (1998)'])
dataClark=dataClark[['14C age in years from Clark et al. (1998)','DistanceRecharge_km']]
dataClark['some_value_color'] = dataClark['14C age in years from Clark et al. (1998)'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ClarkIndex=np.where(~np.isnan(data['14C age in years from Clark et al. (1998)']))


#
dataModelDarcy=data.dropna(subset=['Residence time in years from my estimation using Darcy Law'])
dataModelDarcy=dataModelDarcy[['Residence time in years from my estimation using Darcy Law','DistanceRecharge_km']]
dataModelDarcy['some_value_color'] = dataModelDarcy['Residence time in years from my estimation using Darcy Law'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelDarcyIndex=np.where(~np.isnan(data['Residence time in years from my estimation using Darcy Law']))

#
dataModel=data.dropna(subset=['Residence time in years from the model'])
dataModel=dataModel[['Residence time in years from the model','DistanceRecharge_km']]
dataModel['some_value_color'] = dataModel['Residence time in years from the model'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelIndex=np.where(~np.isnan(data['Residence time in years from the model']))



colorMacFarlane=mcolors.to_hex(mapper.to_rgba(10400))
colorWhittemore=mcolors.to_hex(mapper.to_rgba(210000))
colorThorntonUp=mcolors.to_hex(mapper.to_rgba(3.65e7))
colorThorntonLow=mcolors.to_hex(mapper.to_rgba(365000))
colorHegelLow=mcolors.to_hex(mapper.to_rgba(1.5e5))
colorHegelUp=mcolors.to_hex(mapper.to_rgba(1.5e8))


fig, ax = plt.subplots()#ax = fig.add_subplot(111)

fig1 = ax.barh(ind[NeilIndex]+0.6,dataNeil['DistanceRecharge_km'],width,color=dataNeil.some_value_color.values,label='Neil',hatch='\\')
fig2 = ax.barh(ind[ClarkIndex],dataClark['DistanceRecharge_km'],width,color=dataClark.some_value_color.values,label='Clark')
fig3 = ax.barh(ind[ModelDarcyIndex] +0.2,dataModelDarcy['DistanceRecharge_km'],width,color=dataModelDarcy.some_value_color.values,label='Model Darcy',hatch='|')
fig4 = ax.barh(ind[ModelIndex] +0.4,dataModel['DistanceRecharge_km'],width,color=dataModel.some_value_color.values,label='Model',hatch='/')
fig5 = ax.barh(-1,600,width,color=colorMacFarlane,label='MacFarlane')
fig6 = ax.barh(-2,600,width,color=colorWhittemore,label='WhitteMore')
fig7 = ax.barh(-3,600,width,color=colorThorntonLow,label='Thornton, lower value')
fig8 = ax.barh(-4,600,width,color=colorThorntonUp,label='Thornton, upper value')
fig9 = ax.barh(-5,600,width,color=colorHegelLow,label='Helgessen, lower value')
fig10 = ax.barh(-6,600,width,color=colorHegelUp,label='Helgessen, upper value')




ax.set_yticks(ind)
#ax.set_yticklabels([-1, 'MacFralane'])
ax.set_yticklabels(data.index)
#ax.grid(axis='y')
ax.set_xlabel('Distance from recharge in km')
ax.set_ylabel('Residence time in years')
cbar=plt.colorbar(mapper)
cbar.set_label('Residence time in years')
plt.legend(loc='best', prop={'size': 10})
plt.show()

#%%
fig1 = ax.barh(dataNeil.index,dataNeil['DistanceRecharge_km'],width,color='blue')
#fig2 = ax.barh(data.index,data['Residence time in years from the model'],width)
#fig3 = ax.barh(data.index,data['Residence time in years from my estimation using Darcy Law'],width)
fig4 = ax.barh(dataClark.index,dataClark['DistanceRecharge_km'],width,color='red')

ax.set_xlabel('Distance from recharge in km')
ax.set_ylabel('Residence time in years')
#ax.set_xscale('log')
plt.xticks(rotation='vertical')
plt.legend(loc='best')
plt.show()

#%%
import matplotlib
df = pd.DataFrame(np.random.randint(0,21,size=(7, 2)), columns=['some_value', 'another_value'])
df.iloc[-1] = np.nan 
norm = matplotlib.colors.Normalize(vmin=min(dataModel['Residence time in years from the model'].values), vmax=max(dataModel['Residence time in years from the model'].values), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
dataModel['some_value_color'] = dataModel['Residence time in years from the model'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
dataModel