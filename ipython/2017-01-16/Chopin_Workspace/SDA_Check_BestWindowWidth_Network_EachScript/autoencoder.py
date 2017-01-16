# -*- coding: utf-8 -*-
import numpy as np
import os
# Generate WindowFrame Function
import joblib

from tqdm import tqdm,tqdm_notebook
# FFT library
from scipy import fftpack
from scipy import signal

# Generate WindowFrame Function
from mymodule import window

# timer
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Layer
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers.core import Dropout

def GetWindowFrame(windowName, dataRaw, wWidth, sWidth, PATH):
    """
    data　ウィンドウフレームに変換するデータ
    registName　ウィンドウフレームに登録するデータの名前
    windowWidth　ウィンドウ幅
    slidingWidth　スライド幅
    PATH ウィドウフレームを保存するディレクトリ
    
    
    About Function:
        与えられたdataからウィンドウ幅windowWidth,スライド幅slidingWidthにしたがって
        registNameのウィドウフレームを返す．
        また，与えれたPATH内に同様なパラメータ( WindowWidth, slidingWidth)かつ同様な
        windowNameのものがある場合，そのデータを返す．
        この関数が登録，ウィンドウフレームに変換できるデータは１つとする
    """
    storedName = windowName+'_Win='+str(wWidth).zfill(4)+'_Sld='+str(sWidth).zfill(4)+'.npz'
    l = os.listdir(PATH)
    
    if storedName in l:
        print "this data had finished making"
        return np.load(PATH+storedName)['data'][()]
    
    w=window()
    w.SetData(windowName,dataRaw)    
    wind=w.Compile(wWidth,sWidth)
    windoW=wind.reshape((len(wind),np.prod(wind.shape[1:])))
    
    np.savez(PATH+storedName,data=wind)

    return wind

es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

def StackedAE(x_train,x_test,
              encoding_dim1,encoding_dim2,encoding_dim3,CommonName,):
    """
    x_train,x_test
    windowNum,slidingNum -> Sliding Window parametor
    encoding_dimX -> each layer's dimention
    """
      
    ## First Training
    # this is our input placeholder
    input_img1 = Input(shape=(x_train.shape[1],))
    encoded1 = Dense(encoding_dim1,activation='tanh',
                    W_regularizer=regularizers.WeightRegularizer(l1=0.001)
                    )(input_img1)
    encoder1 = Model(input=input_img1, output=encoded1)
    decoded1 = Dense(x_train.shape[1], activation='linear')(encoded1)
    autoencoder1 = Model(input=input_img1, output=decoded1)
    autoencoder1.compile(optimizer='adadelta', loss='mean_absolute_error')
    autoencoder1.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    validation_data=(x_test, x_test),
                    callbacks=[es_cb],
                    verbose=2)
    #plot(autoencoder1, to_file='AutoEncoder1.png')
    FirstAeOut = encoder1.predict(x_train)

    ## Second Training
    # this is our input placeholder
    input_img2 = Input(shape=(encoding_dim1,))
    encoded2 = Dense(encoding_dim2, activation='tanh',
                    )(input_img2)
    encoder2 = Model(input=input_img2, output=encoded2)
    decoded2 = Dense(encoding_dim1, activation='linear')(encoded2)
    autoencoder2 = Model(input=input_img2, output=decoded2)
    autoencoder2.compile(optimizer='adadelta', loss='mean_absolute_error')
    autoencoder2.fit(FirstAeOut, FirstAeOut,
                    nb_epoch=50,
                    batch_size=256,
                    callbacks=[es_cb],
                    verbose=2)
    #plot(autoencoder2, to_file='AutoEncoder2.png')
    SecondAeOut = encoder2.predict(FirstAeOut)

    ## Third Training
    # this is our input placeholder
    input_img3 = Input(shape=(encoding_dim2,))
    encoded3 = Dense(encoding_dim3, activation='tanh',
                    )(input_img3)
    encoder3 = Model(input=input_img3, output=encoded3)
    decoded3 = Dense(encoding_dim2, activation='linear')(encoded3)
    autoencoder3 = Model(input=input_img3, output=decoded3)
    autoencoder3.compile(optimizer='adadelta', loss='mean_absolute_error')
    autoencoder3.fit(SecondAeOut, SecondAeOut,
                    nb_epoch=50,
                    batch_size=256,
                    callbacks=[es_cb],
                    verbose=2)
    #plot(autoencoder3, to_file='AutoEncoder3.png')
    ThirdAeOut = encoder3.predict(SecondAeOut)

    ## Fine Tuning
    SAE = Sequential()
    SAE.add(encoder1.layers[0])
    SAE.add(encoder1.layers[1])
    SAE.add(encoder2.layers[1])
    SAE.add(encoder3.layers[1])
    SAE.add(autoencoder3.layers[-1])
    SAE.add(autoencoder2.layers[-1])
    SAE.add(autoencoder1.layers[-1])
    SAE.compile(optimizer='adadelta', loss='mean_absolute_error')
    plot(SAE, to_file='AutoEncoder.png')
    return SAE
