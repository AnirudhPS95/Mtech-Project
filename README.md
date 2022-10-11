# Mtech-Project
My thesis project for Mtech on development of ML tool for extraction of modes
Here the following two objectives are presented:

1. Extraction of zonal data containing the crux of the flow field
2. Selection of best performing activation function from the 9 default ones availabel in Keras API
    


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
    To remedy this extraction of zone size (128 X128) containing  main flow was extracted. This was 
achieved using TecPlot. Since, the task of zone extraction can become mundane, it was automated using 
a Macro script (Zone_extraction_macro.mcr).

The Tecplot thus generates required files containing data and also the details. These details have to 
be removed before the file can be read in python using pandas. For this code was written which is given in the
remove_headings.py file
    It is to be noted that the above task was done seperately just for sake of convenience.
    
    
***********************************************************************************
**************************Part 3***************************************************
 
 The average flow field has to be calculated. This is the zeroeth mode of the flow. Calculated using the code 
 given in file
 
 
 
 **********************************************************************************
 *************************Part 4***************************************************
 
 Next the main step is performed, which is training of the MD-CNN-AE. The python file having the full code is given in the
 python file CCNN128avgmdcnn.py. 
 
 The Keras API has 9 default activation functions. All these activation functions were used for performing the training. 
 The result was analysed, mainly it being the error value. Based on that best performing activation was to be selected. 
 The result though was inconclusive and thus same class of data was required to reach at some firm conclusion on best activation 
 function selection. 
 
 For this another periodic data was selected. All steps performed on this dataset is similar to ones mentioned above.  
 
 After the analysis the two contenders were for best activation functions were selected, being Elu and Tanh. 
 
 For complete details on the project refer to the dissertation report. 







