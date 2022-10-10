# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:03:13 2021

@author: Home
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 20:24:36 2021

@author: Home
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:36:11 2021

@author: Home
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:59:28 2021

@author: HP
"""
################ Importin Data############
import glob
import numpy as np
path="*.dat"
file_list=sorted(glob.glob(path))


############# Loading Data in pandas dataframe########
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


############### Loading the average flow field data that is the zeroeth mode
avg=np.loadtxt("avgflow.txt")  




######### Redundant step. Could have used Numpy array
Xarray=Xdf.to_numpy()
Yarray=Ydf.to_numpy()


##### Feature engineering-Excluding the zeroeth of the data
X=np.zeros((16384,180))
Y=np.zeros((16384,180))

for i in range(180):
    X[:,i]=Xarray[:,i]-avg[:,0]
    Y[:,i]=Yarray[:,i]-avg[:,1]
    

### Reshaping the data suitable for use in the CNN-AE
X=np.transpose(X)
Y=np.transpose(Y)

Z=np.array([X,Y])
Z=Z.reshape(2,180,128,128)
X=np.einsum('kijl->ijlk',Z)

#%% Importing libraries for CNN
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split


#%%Encoder construction
input_img = Input(shape=(128,128, 2))
## Encoder
x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(64,64)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(32,32)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(16,16)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(8,8)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(4,4)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(2,2)
x = Reshape([2*2*4])(x)
encoded = Dense(2,activation='tanh')(x)

encoder=Model(input_img,encoded)
encoder.summary()


#%% Mode decompostion part############
######################################
# Courtsey to code given by Murata et. al. 

val1= Lambda(lambda x: x[:,0:1])(encoded)
val2= Lambda(lambda x: x[:,1:2])(encoded)

#%%Decoder construction

x1 = Dense(2*2*4,activation='tanh')(val1)
x1 = Reshape([2,2,4])(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(4,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1d = Conv2D(2,(3,3),activation='linear',padding='same')(x1)

decoded1=x1d
decoder1=Model(input_img,decoded1)
decoder1.summary()

#%%

## Decoder 2
x2 = Dense(2*2*4,activation='tanh')(val2)
x2 = Reshape([2,2,4])(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(4,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(16,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2d = Conv2D(2,(3,3),activation='linear',padding='same')(x2)

decoded2=x2d
decoder2=Model(input_img,decoded2)
decoder2.summary()

############ Adding decoded value obtaine from seperate modes
############ Simple arithemetic will suffice

decoded = Add()([x1d,x2d])

#%%Autoencoder construction
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss="mse", optimizer=Adam(lr=0.001))



############ Training on data

an='./2000epochstanhMD.hdf5'
model_cb=ModelCheckpoint(an, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=50,verbose=1)
log_csv=CSVLogger('2000epochstanhMD.csv',separator=',', append=False)
cb = [model_cb, early_cb, log_csv]

X_train,X_test,y_train,y_test=train_test_split(X,X,test_size=0.055,random_state=1)

history=autoencoder.fit(X_train, y_train,
                epochs=2000,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=cb )

### Extracting the modes in the highest dimension 
mode1=decoder1.predict(X)
mode2=decoder2.predict(X)

### Modes of both velocity
Umode1=mode1[:,:,:,0:1]
Umode2=mode2[:,:,:,0:1]
Vmode1=mode1[:,:,:,1:2]
Vmode2=mode2[:,:,:,1:2]

###  Reshaping the Modes and real velocities 
Umode1=Umode1.reshape(180,16384)
Umode2=Umode2.reshape(180,16384)
Vmode1=Vmode1.reshape(180,16384)
Vmode2=Vmode2.reshape(180,16384)
#%%
Xreal=X
Ureal=Xreal[:,:,:,0:1]
Vreal=Xreal[:,:,:,1:2]

Ureal=Ureal.reshape(180,16384)
Vreal=Vreal.reshape(180,16384)

Urec=Xrec[:,:,:,0:1]
Vrec=Xrec[:,:,:,1:2]

Urec=Urec.reshape(180,16384)
Vrec=Vrec.reshape(180,16384)

Ureal=np.transpose(Ureal)
Vreal=np.transpose(Vreal)
Urec=np.transpose(Urec)
Vrec=np.transpose(Vrec)

#### Adding back the zeroeth mode or average flow field
for i in range(180):
    Ureal[:,i]=Ureal[:,i]+avg[:,0]
    Urec[:,i]=Urec[:,i]+avg[:,0]
    Vreal[:,i]=Vreal[:,i]+avg[:,1]
    Vrec[:,i]=Vrec[:,i]+avg[:,1]
    mode11[:,i]=mode1[:,i]+avg[:,0]
    mode1[:,i]=mode1[:,i]+avg[:,1]