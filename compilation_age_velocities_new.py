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



--------------> New version with new input table / Dakota aquifer only 
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
from textwrap import wrap


#%%

plt.close('all')

os.chdir("C:/Users/Reika/Dropbox/RessourceWaterMidcontinent/Well_Data")


width = 0.3

data = pd.read_csv("compilation_Dakota_well_3-20-20.csv",encoding='cp1252')
data=data.set_index('Well name ')
data=data.sort_values(by=['DistanceRecharge_km'],ascending =False)


## drop well with no information / are not tapping into Dakota aquifer

data = data.drop(['Clay Center ','MRN 1072','MRN 559','MRN 1174','STOCK WELL  LAWRENCE DOOLEY','poky feeders feedlot', 'City of McPherson, Kansas',
       'MRN 1059 = Wilson reservoir, sampled by Richard Schoen (H2O SMELL  TANK CONTAIN RED-COLORED WATER)',
       'MRN 501','Lincoln center '])



N=data.shape[0]
ind = np.arange(data.shape[0])

#clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
#         (0.7, "green"), (0.75, "blue"), (1, "blue")]
#rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
#
#x = np.arange(N).astype(float)

#norm = matplotlib.colors.Normalize(vmin=0.1, vmax=50, clip=True)
norm = matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2, clip=True)
#norm = matplotlib.colors.LogNorm(vmin=min(dataClark['Velocity in m/year based on Clark (1998)'].values), vmax=1.5e8, clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
mapper.set_array(data['Velocity in m/year from my estimation using Darcy Law'].values)


#%% Different plot for each source (model/clark...)

dataNeil=data.dropna(subset=['Velocity in m/year based on Neil'])
dataNeil=dataNeil[['Velocity in m/year based on Neil','DistanceRecharge_km']]
dataNeil['some_value_color'] = dataNeil['Velocity in m/year based on Neil'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
NeilIndex=np.where(~np.isnan(data['Velocity in m/year based on Neil']))

#
dataClark=data.dropna(subset=['Velocity in m/year based on Clark (1998)'])
dataClark=dataClark[['Velocity in m/year based on Clark (1998)','DistanceRecharge_km']]
dataClark['some_value_color'] = dataClark['Velocity in m/year based on Clark (1998)'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ClarkIndex=np.where(~np.isnan(data['Velocity in m/year based on Clark (1998)']))


#
dataModelDarcy=data.dropna(subset=['Velocity in m/year from my estimation using Darcy Law'])
dataModelDarcy=dataModelDarcy[['Velocity in m/year from my estimation using Darcy Law','DistanceRecharge_km']]
dataModelDarcy['some_value_color'] = dataModelDarcy['Velocity in m/year from my estimation using Darcy Law'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelDarcyIndex=np.where(~np.isnan(data['Velocity in m/year from my estimation using Darcy Law']))

#
dataModel=data.dropna(subset=['Velocity in m/year from the model'])
dataModel=dataModel[['Velocity in m/year from the model','DistanceRecharge_km']]
dataModel['some_value_color'] = dataModel['Velocity in m/year from the model'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelIndex=np.where(~np.isnan(data['Velocity in m/year from the model']))



colorMacFarlane=mcolors.to_hex(mapper.to_rgba(27.8))
colorWhittemore=mcolors.to_hex(mapper.to_rgba(1.8))
colorThorntonUp=mcolors.to_hex(mapper.to_rgba(1))
colorThorntonLow=mcolors.to_hex(mapper.to_rgba(0.01))
colorHegelLow=mcolors.to_hex(mapper.to_rgba(0.003))
colorHegelUp=mcolors.to_hex(mapper.to_rgba(3))


fig, ax = plt.subplots()#ax = fig.add_subplot(111)

fig1 = ax.barh(ind[ClarkIndex],dataClark['DistanceRecharge_km'],width,color=dataClark.some_value_color.values,label='14C from Clark et al (1998)',edgecolor='black')
fig2 = ax.barh(ind[ModelIndex] +0.2,dataModel['DistanceRecharge_km'],width,color=dataModel.some_value_color.values,label='Hydrogeological model',hatch='/')
fig3 = ax.barh(ind[NeilIndex]+0.4,dataNeil['DistanceRecharge_km'],width,color=dataNeil.some_value_color.values,label='81Kr from Neil.C',hatch='\\',edgecolor='black')
fig4 = ax.barh(ind[ModelDarcyIndex] +0.6,dataModelDarcy['DistanceRecharge_km'],width,color=dataModelDarcy.some_value_color.values,label='Derived from Darcy''s equation',hatch='|')
fig5 = ax.barh(-2,600,width,color=colorMacFarlane,label='MacFarlane')
fig6 = ax.barh(-3,600,width,color=colorWhittemore,label='WhitteMore')
fig7 = ax.barh(-4,600,width,color=colorThorntonLow,label='Thornton, lower value')
fig8 = ax.barh(-5,600,width,color=colorThorntonUp,label='Thornton, upper value')
fig9 = ax.barh(-6,600,width,color=colorHegelLow,label='Helgessen, lower value')
fig10 = ax.barh(-7,600,width,color=colorHegelUp,label='Helgessen, upper value')


labels=data.index
labels = [ '\n'.join(wrap(l, 29)) for l in labels ]



ax.set_yticks(ind)
#ax.set_yticklabels([-1, 'MacFralane'])
ax.set_yticklabels(labels,fontsize=8)
#ax.grid(axis='y')
ax.set_xlabel('Distance from recharge in km',fontsize=18)
#ax.set_ylabel('Residence time in years',fontsize=18)
cbar=plt.colorbar(mapper)
cbar.set_label('Velocity in m/yr',fontsize=15)
plt.legend(loc='best', prop={'size': 18})
plt.show()

#%%
fig1 = ax.barh(dataNeil.index,dataNeil['DistanceRecharge_km'],width,color='blue')
#fig2 = ax.barh(data.index,data['Velocity in m/year from the model'],width)
#fig3 = ax.barh(data.index,data['Velocity in m/year from my estimation using Darcy Law'],width)
fig4 = ax.barh(dataClark.index,dataClark['DistanceRecharge_km'],width,color='red')

ax.set_xlabel('Distance from recharge in km')
ax.set_ylabel('Velocity in m/yr')
#ax.set_xscale('log')
plt.xticks(rotation='vertical')
plt.legend(loc='best')
plt.show()

#%%
import matplotlib
df = pd.DataFrame(np.random.randint(0,21,size=(7, 2)), columns=['some_value', 'another_value'])
df.iloc[-1] = np.nan 
norm = matplotlib.colors.Normalize(vmin=min(dataModel['Velocity in m/year from the model'].values), vmax=max(dataModel['Velocity in m/year from the model'].values), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
dataModel['some_value_color'] = dataModel['Velocity in m/year from the model'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
dataModel


#%%

#%% on elimine bar pour regionnal flow

dataNeil=data.dropna(subset=['Velocity in m/year based on Neil'])
dataNeil=dataNeil[['Velocity in m/year based on Neil','DistanceRecharge_km']]
dataNeil['some_value_color'] = dataNeil['Velocity in m/year based on Neil'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
NeilIndex=np.where(~np.isnan(data['Velocity in m/year based on Neil']))

#
dataClark=data.dropna(subset=['Velocity in m/year based on Clark (1998)'])
dataClark=dataClark[['Velocity in m/year based on Clark (1998)','DistanceRecharge_km']]
dataClark['some_value_color'] = dataClark['Velocity in m/year based on Clark (1998)'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ClarkIndex=np.where(~np.isnan(data['Velocity in m/year based on Clark (1998)']))


#
dataModelDarcy=data.dropna(subset=['Velocity in m/year from my estimation using Darcy Law'])
dataModelDarcy=dataModelDarcy[['Velocity in m/year from my estimation using Darcy Law','DistanceRecharge_km']]
dataModelDarcy['some_value_color'] = dataModelDarcy['Velocity in m/year from my estimation using Darcy Law'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelDarcyIndex=np.where(~np.isnan(data['Velocity in m/year from my estimation using Darcy Law']))

#
dataModel=data.dropna(subset=['Velocity in m/year from the model'])
dataModel=dataModel[['Velocity in m/year from the model','DistanceRecharge_km']]
dataModel['some_value_color'] = dataModel['Velocity in m/year from the model'].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
ModelIndex=np.where(~np.isnan(data['Velocity in m/year from the model']))



colorMacFarlane=mcolors.to_hex(mapper.to_rgba(27.8))
colorWhittemore=mcolors.to_hex(mapper.to_rgba(1.8))
colorThorntonUp=mcolors.to_hex(mapper.to_rgba(1))
colorThorntonLow=mcolors.to_hex(mapper.to_rgba(0.01))
colorHegelLow=mcolors.to_hex(mapper.to_rgba(0.003))
colorHegelUp=mcolors.to_hex(mapper.to_rgba(3))


fig, ax = plt.subplots()#ax = fig.add_subplot(111)

fig1 = ax.barh(ind[ClarkIndex],dataClark['DistanceRecharge_km'],width,color=dataClark.some_value_color.values,label='14C from Clark et al (1998)',edgecolor='black')
fig2 = ax.barh(ind[ModelIndex] +0.2,dataModel['DistanceRecharge_km'],width,color=dataModel.some_value_color.values,label='Hydrogeological model',hatch='/')
fig3 = ax.barh(ind[NeilIndex]+0.4,dataNeil['DistanceRecharge_km'],width,color=dataNeil.some_value_color.values,label='81Kr from Neil.C',hatch='\\',edgecolor='black')
fig4 = ax.barh(ind[ModelDarcyIndex] +0.6,dataModelDarcy['DistanceRecharge_km'],width,color=dataModelDarcy.some_value_color.values,label='Derived from Darcy''s equation',hatch='|')
fig5 = ax.scatter(-2,600,width,color=colorMacFarlane,label='MacFarlane')
fig6 = ax.barh(-3,600,width,color=colorWhittemore,label='WhitteMore')
fig7 = ax.barh(-4,600,width,color=colorThorntonLow,label='Thornton, lower value')
fig8 = ax.barh(-5,600,width,color=colorThorntonUp,label='Thornton, upper value')
fig9 = ax.barh(-6,600,width,color=colorHegelLow,label='Helgessen, lower value')
fig10 = ax.barh(-7,600,width,color=colorHegelUp,label='Helgessen, upper value')


labels=data.index
labels = [ '\n'.join(wrap(l, 29)) for l in labels ]



ax.set_yticks(ind)
#ax.set_yticklabels([-1, 'MacFralane'])
ax.set_yticklabels(labels,fontsize=8)
#ax.grid(axis='y')
ax.set_xlabel('Distance from recharge in km',fontsize=18)
#ax.set_ylabel('Residence time in years',fontsize=18)
cbar=plt.colorbar(mapper)
cbar.set_label('Velocity in m/yr',fontsize=15)
plt.legend(loc='best', prop={'size': 18})
plt.show()
