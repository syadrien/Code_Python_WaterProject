# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:17:42 2020

@author: Adrien


Plot of agragation of data 
Data on salinity Na, Cl, Br
+18o and dD

Data from Clark, Scheerhorn and http://www.kgs.ku.edu/Dakota/vol2/vol2wat.htm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

plt.close('all')

os.chdir("C:/Users/Reika/Dropbox/RessourceWaterMidcontinent/Well_Data")
# C:/Users/Reika/Dropbox/RessourceWaterMidcontinent/quality_dakota


data = pd.read_csv("AgragationDataTable_v3.csv")

plt.plot(data['Cl in mg/L'],data['Br in mg/L']/data['Cl in mg/L']*1e4,'c.',label='All Data', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='GP1'],data['Br in mg/L'][data['classiWell']=='GP1']/data['Cl in mg/L'][data['classiWell']=='GP1']*1e4,'mo',label='GP1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='GP2'],data['Br in mg/L'][data['classiWell']=='GP2']/data['Cl in mg/L'][data['classiWell']=='GP2']*1e4,'ro',label='GP2', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 1'],data['Br in mg/L'][data['classiWell']=='Clark 1']/data['Cl in mg/L'][data['classiWell']=='Clark 1']*1e4,'gs',label='Clark1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 2'],data['Br in mg/L'][data['classiWell']=='Clark 2']/data['Cl in mg/L'][data['classiWell']=='Clark 2']*1e4,'bs',label='Clark2', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 3'],data['Br in mg/L'][data['classiWell']=='Clark 3']/data['Cl in mg/L'][data['classiWell']=='Clark 3']*1e4,marker='s',color='orange',linestyle='None',label='Clark3', markersize=12)


#plot of a couple of reference points
plt.plot(22500,32/22500*1e4,'*k',label='Oilfield water', markersize=15)
plt.plot(150000,17/150000*1e4,'sk',label='Na-Cl brine', markersize=15)
plt.plot(1.98E+04,69/1.98E+04*1e4,'+k',label='Seawater', markersize=15)
plt.plot(120,0.21/120*1e4,'pk',label='South Fork Solomon River (FY92)', markersize=15)
#plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)


plt.rcParams.update({'font.size': 18})

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Cl (in mg/L)',fontsize=18)
plt.ylabel('Br/Cl *10^4 (mass ratio)',fontsize=18)
plt.legend()
plt.grid()

#%%
plt.close('all')


plt.plot(data['Cl in mg/L'],data['Br in mg/L'],'c.',label='All Data', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='GP1'],data['Br in mg/L'][data['classiWell']=='GP1'],'mo',label='GP1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='GP2'],data['Br in mg/L'][data['classiWell']=='GP2'],'ro',label='GP2', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 1'],data['Br in mg/L'][data['classiWell']=='Clark 1'],'gs',label='Clark1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 2'],data['Br in mg/L'][data['classiWell']=='Clark 2'],'bs',label='Clark2', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 3'],data['Br in mg/L'][data['classiWell']=='Clark 3'],marker='s',color='orange',linestyle='None',label='Clark3', markersize=12)

plt.plot(22500,32,'*k',label='Oilfield water', markersize=15)
plt.plot(150000,17,'sk',label='Na-Cl brine', markersize=15)
plt.plot(1.98E+04,69,'+k',label='Seawater', markersize=15)
plt.plot(120,0.21,'pk',label='South Fork Solomon River (FY92)', markersize=15)

#plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)

plt.plot([10340,107800,183300,187900,190500,224000,254000],[35,396,1010,2670,2970,4770,6060],'k.-',label='SET (Carpenter, 1978)')


plt.rcParams.update({'font.size': 18})

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Cl (in mg/L)',fontsize=18)
plt.ylabel('Br (in mg/L)',fontsize=18)
plt.legend()
plt.grid()

#%%

plt.close('all')


plt.plot(data['Cl in mg/L'],data.sodiummgperl,'c.',label='All Data', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='GP1'],data.sodiummgperl[data['classiWell']=='GP1'],'mo',label='GP1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='GP2'],data.sodiummgperl[data['classiWell']=='GP2'],'ro',label='GP2', markersize=12)


plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 1'],data.sodiummgperl[data['classiWell']=='Clark 1'],'gs',label='Clark1', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 2'],data.sodiummgperl[data['classiWell']=='Clark 2'],'bs',label='Clark2', markersize=12)
plt.plot(data['Cl in mg/L'][data['classiWell']=='Clark 3'],data.sodiummgperl[data['classiWell']=='Clark 3'],marker='s',color='orange',linestyle='None',label='Clark3', markersize=12)

#plt.plot(22500,32,'*k',label='Oilfield water', markersize=15)
#plt.plot(150000,17,'sk',label='Na-Cl brine', markersize=15)
#plt.plot(1.98E+04,69,'+k',label='Seawater', markersize=15)
##plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)

x = [1, 100, 10000, 10000,100000]
y = [1, 100, 10000, 10000,100000]

plt.plot(x, y, 'k--')

plt.rcParams.update({'font.size': 22})

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Cl (in mg/L)',fontsize=18)
plt.ylabel('Na (in mg/L)',fontsize=18)
plt.legend()
plt.grid()

#%% oxygen 18 + classification 

plt.close('all')


plt.plot(data['deltaoxygen1618 in ppm'],data.deuteriuminppm,'c.',label='All Data', markersize=12)


plt.plot(data['deltaoxygen1618 in ppm'][data['classiWell']=='GP1'],data.deuteriuminppm[data['classiWell']=='GP1'],'mo',label='GP1', markersize=12)
plt.plot(data['deltaoxygen1618 in ppm'][data['classiWell']=='GP2'],data.deuteriuminppm[data['classiWell']=='GP2'],'ro',label='GP2', markersize=12)


plt.plot(data['deltaoxygen1618 in ppm'][data['classiWell']=='Clark 1'],data.deuteriuminppm[data['classiWell']=='Clark 1'],'gs',label='Clark1', markersize=12)
plt.plot(data['deltaoxygen1618 in ppm'][data['classiWell']=='Clark 2'],data.deuteriuminppm[data['classiWell']=='Clark 2'],'bs',label='Clark2', markersize=12)
plt.plot(data['deltaoxygen1618 in ppm'][data['classiWell']=='Clark 3'],data.deuteriuminppm[data['classiWell']=='Clark 3'],marker='s',color='orange',label='Clark3', markersize=12,linestyle='None')

#plt.plot(22500,32,'*k',label='Oilfield water', markersize=15)
#plt.plot(150000,17,'sk',label='Na-Cl brine', markersize=15)
#plt.plot(1.98E+04,69,'+k',label='Seawater', markersize=15)
##plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)

x = np.arange(-15,-2)
y = [8*i+10 for i in np.arange(-15,-2)]


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
"δ18O [‰]".translate(SUP)


plt.plot(x, y, 'k-',label='Meteoric water line')

plt.rcParams.update({'font.size': 22})

plt.xlabel("δ18O [‰]".translate(SUP),fontsize=22)
#plt.xlabel('δ18O [‰]',fontsize=22)
plt.ylabel('δD [‰]',fontsize=22)
plt.legend()
plt.grid()

#%% oxugen 18 vs D + salinity marker size


plt.close('all')

#data.replace([np.inf, -np.inf], np.nan)
dataBis=data.dropna(subset=['TDS in mg/L'])

fresh=dataBis[dataBis['TDS in mg/L']<=1000]
salt=dataBis[dataBis['TDS in mg/L']>=1000]

fig, ax = plt.subplots()

plt.scatter(fresh['deltaoxygen1618 in ppm'],fresh.deuteriuminppm,s=np.log(fresh['TDS in mg/L']).values**3,label='Freshwater',c='blue')
plt.scatter(salt['deltaoxygen1618 in ppm'],salt.deuteriuminppm,s=np.log(salt['TDS in mg/L']).values**3,label='Saltwater',c='red')

y = [-6,-6,-6,-6]
z = [-80,-90,-100,-110]
size=[100,1000,10000,100000]
n = ['   TDS = 100 mg/L','   TDS = 1,000 mg/L', '   TDS = 10,000 mg/L', '   TDS = 100,000 mg/L']


for i, txt in enumerate(n):
    plt.scatter(y[i], z[i],s=np.log(size[i])**3,c='black')
    plt.annotate(txt, (y[i], z[i]))




x = np.arange(-15,-2)
y = [8*i+10 for i in np.arange(-15,-2)]

plt.plot(x, y, 'k-',label='Meteoric water line')


plt.rcParams.update({'font.size': 22})




plt.xlabel("δ18O [‰]".translate(SUP),fontsize=22)
#plt.xlabel('δ18O [‰]',fontsize=22)
plt.ylabel('δD [‰]',fontsize=22)
plt.legend()
plt.grid()
plt.show()



#%% oxygen 18 + distinction well saple

plt.close('all')


plt.plot(data['deltaoxygen1618 in ppm'],data.deuteriuminppm,'k.',label='All Data', markersize=12)


plt.plot(data['deltaoxygen1618 in ppm'][data['CanSample']=='Yes'],data.deuteriuminppm[data['CanSample']=='Yes'],'ro',label='Can Sample', markersize=12)


#plt.plot(22500,32,'*k',label='Oilfield water', markersize=15)
#plt.plot(150000,17,'sk',label='Na-Cl brine', markersize=15)
#plt.plot(1.98E+04,69,'+k',label='Seawater', markersize=15)
##plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)

x = np.arange(-15,-2)
y = [8*i+10 for i in np.arange(-15,-2)]


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
"δ18O [‰]".translate(SUP)


plt.plot(x, y, 'k-',label='Meteoric water line')

plt.rcParams.update({'font.size': 22})

plt.xlabel("δ18O [‰]".translate(SUP),fontsize=22)
#plt.xlabel('δ18O [‰]',fontsize=22)
plt.ylabel('δD [‰]',fontsize=22)
plt.legend()
plt.grid()


#%% plot Br vs Cl including sampling sites 

plt.close('all')


plt.plot(data['Cl in mg/L'],data['Br in mg/L'],'c.',label='All Data', markersize=12)


plt.plot(data['Cl in mg/L'][data['CanSample']=='Yes'],data['Br in mg/L'][data['CanSample']=='Yes'],'ro',label='Can Sample', markersize=12)

plt.plot(22500,32,'*k',label='Oilfield water', markersize=15)
plt.plot(150000,17,'sk',label='Na-Cl brine', markersize=15)
plt.plot(1.98E+04,69,'+k',label='Seawater', markersize=15)
plt.plot(120,0.21,'pk',label='South Fork Solomon River (FY92)', markersize=15)

#plt.plot(0.2,0.15/0.2,'Dk',label='Freshwater', markersize=15)

plt.plot([10340,107800,183300,187900,190500,224000,254000],[35,396,1010,2670,2970,4770,6060],'k.-',label='SET (Carpenter, 1978)')


plt.rcParams.update({'font.size': 18})

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Cl (in mg/L)',fontsize=18)
plt.ylabel('Br (in mg/L)',fontsize=18)
plt.legend()
plt.grid()
