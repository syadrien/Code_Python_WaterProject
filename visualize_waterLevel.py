# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:42:04 2020

@author: Adrien

Goal : Visualize data availability + time-serie of water well data

Input : Water data from USGS for wells tapping into the Dakota aquifer (filter for Dakota sandstone or formation group [210DKOT]) in a csv table
https://nwis.waterdata.usgs.gov/ks/nwis/gwlevels    

Visualize in the form of a timeline

    
    
"""

#%%  FRISE CHRONOLOGIQUE DISPONIBILITE DONEES
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%


plt.close('all')
os.chdir("C:/Users/elira/Dropbox/RessourceWaterMidcontinent/Water_level_Dakota")

dataRaw=pd.read_csv('gwlevels(1).csv', sep=',')
dataRaw.lev_dt=pd.DatetimeIndex(dataRaw.lev_dt)
dataRaw['Year']=dataRaw.lev_dt.year

dataRawbis=dataRaw.set_index('site_no')
listStation=np.unique(dataRawbis.index)


meandataRawbis=dataRawbis.groupby(dataRawbis.index).mean()

countData=dataRawbis.groupby(dataRawbis.index).count()


dataRawbis['year']=pd.DatetimeIndex(dataRaw.lev_dt).year


j=1
CompteurAn=[];
compilMesureparAn=pd.DataFrame(index=np.arange(1988,2017), columns=listStation)


fig, ax =plt.subplots()
#parcours les fichiers contenus dans le dossier DonneDebitMoyenJournalier
for i in listStation :
    
    nom=i 
    donneStationCourante=dataRawbis.loc[dataRawbis.index == i]

    

#    DataRogne=pd.dataRawbis(index=Tmesure)                        
        
    donneStationCourante['Mesure'] = donneStationCourante.lev_va;
    donneStationCourante.index = pd.to_datetime(donneStationCourante.lev_dt)
    
  #compte le nombre de mesure par mois/an pour le periode ou ya des donnees (groupe par annee + compte)
# pour compte par an     
    MesureParAn=donneStationCourante.groupby(donneStationCourante.index.year).count();

#    donneStationCourante['Mesure']

    print(MesureParAn)
    numero=np.ones(np.size(MesureParAn.index))
    
        
    #recupere info sur Bon/Mauvaise stations pour laps de temps en fonction du critere defini precedemment
    #trick pour scatter.plot avec couleur differente
    

      
    plt.scatter(MesureParAn.index.values,numero*j,s=MesureParAn.Mesure*30,c='b',marker='s')
#    plt.scatter(MauvaisAn,numero*j,s=MesureParAn.Mesure,c='r',marker='s')
    plt.show()
    
#    listStation.append(i);
    
    compilMesureparAn[i] = MesureParAn
                      
    j=j+1
    

ax.minorticks_on()
ax.grid(b=True,which='major', linestyle='-',axis='x', linewidth='0.5', color='gray')
plt.yticks(range(1,j),listStation)
plt.xlabel('Year',fontsize=20)  
plt.ylabel('Station Code',fontsize=20)
plt.title('Number of water level measurement by year \n (point size relate to the number of data available for each year, from 1 to 365)',fontsize=25)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(True)
plt.show()
    


#extraticks=np.arange(1976,2017,7)
#plt.xticks(list(plt.xticks()[0]) + extraticks)
#NbreAnneValide=Counter(CompteurAnbis).most_common();
#Anne=[i[0] for i in NbreAnneValide]
#OccurenceAnne=[i[1] for i in NbreAnneValide]
#
#plt.figure(2)
#plt.bar(Anne,OccurenceAnne)
#plt.xlabel('Annee')
#plt.ylabel('Nombre total d"annee valide')
#%% figure, evolution en fonction du temps pour toutes les stations 

TableStation = pd.DataFrame(listStation,columns = ['StationID'])
TableStation['AquiferDeclineFeet']=np.nan
TableStation['AquiferDecline%of1st']=np.nan
TableStation['FirstYear']=np.nan
TableStation['lastYear']=np.nan



for i in listStation :
    
    nom=i 
    donneStationCourante=dataRawbis.loc[dataRawbis.index == i]

    donneStationCourante['Mesure'] = donneStationCourante.lev_va;
    donneStationCourante.index = pd.to_datetime(donneStationCourante.lev_dt)

    plt.plot(donneStationCourante.index,donneStationCourante.Mesure,'o-')
    
    localisation=TableStation.loc[TableStation.StationID == i].index[0]
    TableStation['AquiferDeclineFeet'][localisation]=donneStationCourante.Mesure[-1] - donneStationCourante.Mesure[0]
    TableStation['AquiferDecline%of1st'][localisation]=(donneStationCourante.Mesure[-1] - donneStationCourante.Mesure[0])/donneStationCourante.Mesure[0]*100

    TableStation['FirstYear'][localisation]=min(donneStationCourante.index.year)
    TableStation['lastYear'][localisation]=max(donneStationCourante.index.year)

plt.xlabel('Time',fontsize=20)  
plt.ylabel(' Water-level value in feet below land surface ',fontsize=20)

#%%
########### Raw quantification of aquifer decline
# for each station, we take the difference between the last (most recent) and 1st measured value of water level
# then make a histogram of the difference

TableStation=TableStation.dropna()

plt.figure()
plt.hist(TableStation.AquiferDeclineFeet,bins='auto', facecolor='g', alpha=0.75)
plt.ylabel(" # of occurence",fontsize=20)
plt.xlabel('Change in water level in feet',fontsize=20)

plt.figure()
plt.hist(TableStation['AquiferDecline%of1st'],bins='auto', facecolor='b', alpha=0.75)
plt.ylabel(" # of occurence",fontsize=20)
plt.xlabel('% Change in water level relative to 1st measure',fontsize=20)