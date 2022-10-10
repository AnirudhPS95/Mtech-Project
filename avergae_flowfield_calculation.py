# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 20:04:22 2021

@author: Home
"""

# Importing data 

import glob
import numpy as np
path="*.dat"
file_list=sorted(glob.glob(path))

#%% Creating a dataframe out of the data
import pandas as pd
y=np.loadtxt(file_list[0])
Xdf=pd.DataFrame(y[:,2])
Ydf=pd.DataFrame(y[:,3])
y=0
i=0
#%%
for file in file_list:
    y=np.loadtxt(file)
    Xdf[i]=pd.DataFrame(y[:,2])
    Ydf[i]=pd.DataFrame(y[:,3])
    y=0
    i=i+1
    


#%% Converting to array for easier calculation
# Also, to work with arrays. Although same can be done with DataFrame
Xarray=Xdf.to_numpy()
Yarray=Ydf.to_numpy()


#%%Average flow field calcuation
i=0
Uavg=np.zeros((16384,1))
Vavg=np.zeros((16384,1))
for j in range(0,16384):
    for i in range(1,179):
        Uavg[j,0]=Uavg[j,0]+2*Xarray[j,i]
        Vavg[j,0]=Vavg[j,0]+2*Yarray[j,i]
    Uavg[j,0]=(Uavg[j,0]+Xarray[j,0]+Xarray[j,179])/360
    Vavg[j,0]=(Vavg[j,0]+Yarray[j,0]+Yarray[j,179])/360