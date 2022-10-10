# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 01:17:40 2021

@author: Home
"""
# Importing data files created by Tecplot
import glob
import numpy as np
path="*.dat"
i=1
data=[]
file_list=sorted(glob.glob(path))


#%% Making new files without the details given by tecplot
# Also renaming files to have them appear in ascending order

for file in file_list[99:180]:
    f=open('D'+str(i)+'.dat',"x")
    with open(file) as input_data:
        for line in input_data:
            data.append(line)
    data1=data[10:]   
    for element in data1:
        f.write(element)
    f.close()
    i+=1
    data=[]
    



