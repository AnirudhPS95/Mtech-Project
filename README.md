# Mtech-Project
My thesis project for Mtech on development of ML tool for extraction of modes


**********************************************************************************
************************DESCRIPTION***********************************************

The work done here was to extract for vortex induced vibtration periodic flow. 
Specifications of flow: 
1. 2D
2. Re = 100
3. Periodic

Some macro specifications of simulation
1. Circular grid size 375 X 305
2. DNS
3. 180 time snaps data gathered


The raw data file contains U, V velocity fields. One time snap raw data is given in SPAC001.dat file 
for reference

The tool used to extract modes for the data is Mode decomposition-CNN-AE, developed by Murata et. al. 



**********************************************************************************
****************************Part 1************************************************

The raw data U and V velocity fields were given for training with MD-CNN-AE. Following files contain the 
results for the work




**********************************************************************************
***************************Part 2*************************************************

As evident the results for above are very poor. The reconstruction error though low , the reconstruction 
through the neural network is quiet poor. This was because the domain for flow field is quiet big and
thus, the MD-CNN-AE is not able to capture the details. 
    To remedy this extraction of zone containing flow was extracted. This was achieved using TecPlot. Since, 
The task of zone extraction can become mundane, it was automated using a Macro script (Zone_extraction_macro.mcr)




