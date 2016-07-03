# coding: utf-8
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy import signal
import time
import os
import matplotlib.pyplot as plt
import pickle
import copy
import time

def ImportCSV(csv_file,freq,SensorName,mode='Round'):
    '''
    csv_file -> ファイル名 mode Round -> 四捨五入
    Roundup -> 切り上げRounddown -> 切り捨て
    '''
    # data dictionary 
    RawData={}   

    # design dataframe and import csv
    data = pd.read_csv(csv_file)
    data.columns=[u'Type',u'Time',u'AccX',u'AccY',u'AccZ',u'GyrX',u'GyrY',u'GyrZ']
    data = data[ data['Type']=='ags']

    # convert numpy.darray 
    # Acc Data  [0.1mG]=>[G]
    # Gyr Data  [0.01dps]=>[dps]   ...dps=degree per second
    AccX=data.AccX.values*0.0001
    AccY=data.AccY.values*0.0001
    AccZ=data.AccZ.values*0.0001
    GyrX=data.GyrX.values*0.01
    GyrY=data.GyrY.values*0.01
    GyrZ=data.GyrZ.values*0.01

    # regist each raw data 
    RawData['AccX'] = AccX
    RawData['AccY'] = AccY
    RawData['AccZ'] = AccZ
    RawData['GyrX'] = GyrX
    RawData['GyrY'] = GyrY
    RawData['GyrZ'] = GyrZ
    RawData['Name'] = SensorName

    # import time by using numpy
    time = data.Time.values #時間の列だけを抽出       

    if mode == 'Roundup':
        func = lambda x: int(x/freq)*freq
    elif mode == 'Rounddown':
        func = lambda x: int(x/freq)*freq
    elif mode == 'Round':
        func = lambda x: int((x+freq/2)/freq)*freq
    #ERROR
    else:
        print 'check mode and inputed word is caused error'
        return -1

    output = map(func,time)
    RawData['Time'] = np.array(output)

    return RawData

def CalcStartTime(array):
    MAX = min(array[0])
    
    for i in range(len(array)):
        if MAX < min(array[i]):
            MAX = min(array[i])
     
    return  MAX

def CalcGoalTime(array):
    MIN = max(array[0])
    
    for i in range(len(array)):
        if MIN > max(array[i]):
            MIN = max(array[i])
     
    return  MIN

def NanPating(DicData,freq):
    """
    checkData に入れるものは辞書型にする
    freqは計測周期
    """
    start_time = time.time()
    
    # detection for hidden Nan Data
    diffNum =np.array([])
    diffIndex=np.array([])
    checkData = DicData['Time']
    width = len(checkData)
    for i in range(0,width-1):
        if ( checkData[i+1]-checkData[i] )!=freq:
            diffNum=np.append(diffNum, int(checkData[i+1]-checkData[i]) )
            diffIndex=np.append(diffIndex,i)
   
    # insert NAN data to SensorData
    # insert time_data
    def Insert(data,dI,dN,f,mode):
        StartIndex= 0
        tmp =np.array([])
        if mode =='Sensor':
            # insert NAN DATA
            adding = np.nan
            for count,l in enumerate(dI):
                tmp = np.append(tmp, data[StartIndex:int(l)])
                for i in range(0,int(dN[count]/f) ):
                    tmp = np.append(tmp,np.nan)
                StartIndex = int(l)+1
            tmp=np.append(tmp, data[StartIndex:])
        elif mode =='Time':
            # insert 
            for count,l in enumerate(dI):
                tmp = np.append(tmp, data[StartIndex:int(l)])
                for i in range(0,int(dN[count]/f) ):
                    t = int( tmp[-1]+f )
                    tmp = np.append(tmp,t)
                StartIndex = int(l)+1
            tmp=np.append(tmp, data[StartIndex:])
        else:
            print 'mode name error'
        return tmp
    Array ={}
    tmpArrayAccX=Insert(DicData['AccX'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayAccY=Insert(DicData['AccY'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayAccZ=Insert(DicData['AccZ'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayGyrX=Insert(DicData['GyrX'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayGyrY=Insert(DicData['GyrY'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayGyrZ=Insert(DicData['GyrZ'],diffIndex,diffNum,freq,mode='Sensor')
    tmpArrayTime=Insert(DicData['Time'],diffIndex,diffNum,freq,mode='Time')
    Array['AccX'] = tmpArrayAccX
    Array['AccY'] = tmpArrayAccY
    Array['AccZ'] = tmpArrayAccZ
    Array['GyrX'] = tmpArrayGyrX
    Array['GyrY'] = tmpArrayGyrY
    Array['GyrZ'] = tmpArrayGyrZ
    Array['Time'] = tmpArrayTime
    Array['Name'] = DicData['Name']
    #Array=[Time:tmpArrayTime,tmpArrayAccX,tmpArrayAccY,tmpArrayAccZ,tmpArrayGyrX,tmpArrayGyrY,tmpArrayGyrZ]
    elapsed_time = time.time() -start_time
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
    return Array

def CalcStartTime(dic):
    """
    dic　辞書型のリスト
    """
    MAX = min(dic[0]['Time'])
    
    for i in range(len(dic)):
        if MAX < min(dic[i]['Time']):
            MAX = min(dic[i]['Time'])
     
    return  MAX

def CalcGoalTime(dic):
    """
    dic　辞書型のリスト
    """
    MIN = max(dic[0]['Time'])
    
    for i in range(len(dic)):
        if MIN > max(dic[i]['Time']):
            MIN = max(dic[i]['Time'])
     
    return  MIN

def CalcSearchIndexFromTime(data, keyTime):
    """
    data　辞書型
    keyTime data['Time']の中の探す値
    """
    count = 0
    for i in range(0, len(data['Time'])):
        if keyTime == data['Time'][i]:
            print str(keyTime)+' is much in the index  whose number is '+str(i)
            return i
        
# name name.pickle = reservedName 
def SaveDicDataFromFileNPZ(PATH,name,data):
    if not ( os.path.exists(PATH) ): os.makedirs(PATH)

    np.savez(PATH+name, data=data)

def LoadDicDataFromFileNPZ(loadName):
    """
    loadName ロードするファイル名
    """
    root, ext = os.path.splitext(loadName)
    if len(ext) == 0:
        arrays = np.load(loadName+'.npz')
    else:
        arrays = np.load(loadName)

    output = arrays['data'][()]
    return output

def MakeCommonSection(inputDataArray):
    # 共通区間のスタート時間、ゴール時間を求める
    startTime = CalcStartTime(inputDataArray)
    goalTime = CalcGoalTime(inputDataArray)

    # 共通区間のスタート時間のインデックス、ゴール時間のインデックスを探索する
    startIndex = np.array([])
    goalIndex = np.array([])
    for obj in inputDataArray:
        print 'start'
        startIndex = np.append(startIndex, CalcSearchIndexFromTime(obj, startTime) ).astype(int)
        print 'goal'
        goalIndex = np.append(goalIndex, CalcSearchIndexFromTime(obj, goalTime) ).astype(int)

    tmp={}
    comDataArray =[]
    key={}

    # センサデータすべて（時刻、加速度、角速度）に対して共通区間のみのデータを抽出
    for number,iDA in enumerate( inputDataArray ):

        tmp['AccX'] = copy.deepcopy( iDA['AccX'][startIndex[number]:goalIndex[number]] )
        tmp['AccY'] = copy.deepcopy( iDA['AccY'][startIndex[number]:goalIndex[number]] )
        tmp['AccZ'] = copy.deepcopy( iDA['AccZ'][startIndex[number]:goalIndex[number]] )
        tmp['GyrX'] = copy.deepcopy( iDA['GyrX'][startIndex[number]:goalIndex[number]] )
        tmp['GyrY'] = copy.deepcopy( iDA['GyrY'][startIndex[number]:goalIndex[number]] )
        tmp['GyrZ'] = copy.deepcopy( iDA['GyrZ'][startIndex[number]:goalIndex[number]] )
        tmp['Time'] = copy.deepcopy( iDA['Time'][startIndex[number]:goalIndex[number]] )
        tmp['Name'] = copy.deepcopy( iDA['Name'] )
        comDataArray.append(copy.deepcopy(tmp) )    
        key[ tmp['Name'] ] = number
        
        # restore dictionary data
        if not ( os.path.exists(RestorePath) ): os.makedirs(RestorePath)
        SaveDicDataFromFileNPZ(RestorePath+tmp['Name'],tmp)
        
    return key,comDataArray

def GetDicDataFromList(key,dicList,search):
    """
    key dicListの配列番号とdicの名前をつなげる配列
    dicList 辞書型のデータをまとめたリスト
    search 検索するセンサの名前
    """
    i = key[search]
    return dicList[i]

def Dic2darray(dicData):
    tmp = np.array([])
    
    keys = dicData.keys()
    keys.remove('Name')
    output = dicData[keys[0]]
    for i in range(1, len(keys) ):
        tmp=dicData[keys[i]]
        output = np.vstack( (output,tmp) )
    return output.T

def PopWindow(dicData,window,overlap=0.5):
    """
    前提条件として、dicDataのあるインデックスiと次のインデックスi+1との差はサンプリング周期
    もう少し、この関数は変更する
    まずはセンサデータ１つに対してwindowを作るための関数
    """
    data = dic2darray(dicData)
    i=0
    rows,cols = data.shape
    data = data[0:rows-(rows%window),:cols]
    yield data[i:i+window,:]
    i += int( window*overlap )
    yield data[i:i+window,:]