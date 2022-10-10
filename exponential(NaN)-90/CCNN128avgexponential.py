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

import glob
import numpy as np
path="*.dat"
file_list=sorted(glob.glob(path))


#%%
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
#%%
avg=np.loadtxt("avgflow.txt")  




#%%
Xarray=Xdf.to_numpy()
Yarray=Ydf.to_numpy()
#%%
X=np.zeros((16384,180))
Y=np.zeros((16384,180))

for i in range(180):
    X[:,i]=Xarray[:,i]-avg[:,0]
    Y[:,i]=Yarray[:,i]-avg[:,1]
    

#%%
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
input_img = Input(shape=(128, 128, 2))
## Encoder
x = Conv2D(16, (3, 3), activation='exponential', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(64,64)
x = Conv2D(8, (3, 3), activation='exponential', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(32,32)
x = Conv2D(8, (3, 3), activation='exponential', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(16,16)
x = Conv2D(8, (3, 3), activation='exponential', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(8,8)
x = Conv2D(4, (3, 3), activation='exponential', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(4,4)
x = Conv2D(4, (3, 3), activation='exponential', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)                             #(2,2)
x = Reshape([2*2*4])(x)
encoded = Dense(2,activation='exponential')(x)

encoder=Model(input_img,encoded)
encoder.summary()

#%%Decoder construction

decoder_input = Input(shape=(2,), name="decoder_input")
x1 = Dense(2*2*4,activation='exponential')(decoder_input)
x1 = Reshape([2,2,4])(x1)                                               #(2,2)
x1 = UpSampling2D((2,2))(x1)                                            #(4,4)                                        
x1 = Conv2D(4,(3,3),activation='exponential',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)                                            #(8,8)
x1 = Conv2D(8,(3,3),activation='exponential',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)                                            #(16,16)
x1 = Conv2D(8,(3,3),activation='exponential',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)                                            #(32,32)
x1 = Conv2D(8,(3,3),activation='exponential',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)                                            #(64,64)
x1 = Conv2D(16,(3,3),activation='exponential',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)                                            #(128,128)
x1 = Conv2D(2,(3,3),activation='linear',padding='same')(x1)

decoder=Model(decoder_input,x1)
decoder.summary()

#%%Autoencoder construction

ae_input = Input(shape=(128,128,2), name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = Model(ae_input, ae_decoder_output, name="AE")

ae.summary()
ae.compile(loss="mse", optimizer=Adam(lr=0.001))

#%%

an='./2000epochsrexponential.hdf5'
model_cb=ModelCheckpoint(an, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=50,verbose=1)
log_csv=CSVLogger('2000epochsrexponential.csv',separator=',', append=False)
cb = [model_cb, early_cb, log_csv]

X_train,X_test,y_train,y_test=train_test_split(X,X,test_size=0.055,random_state=1)

history=ae.fit(X_train, y_train,
                epochs=2000,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=cb )

#%%
Xrec=ae.predict(X)

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

#%%
for i in range(180):
    Ureal[:,i]=Ureal[:,i]+avg[:,0]
    Urec[:,i]=Urec[:,i]+avg[:,0]
    Vreal[:,i]=Vreal[:,i]+avg[:,1]
    Vrec[:,i]=Vrec[:,i]+avg[:,1]

#%%
y=np.loadtxt(file_list[0])
Xspace=y[:,0:2]
#%%Error 
Uerror=np.zeros((16384,1))
for i in range(16384):
    Uerror[i,:]=np.linalg.norm(Ureal[i,89:90]-Urec[i,89:90])

