"""
    how to use class Window
    1. Regist data using SetData method
        e.g window.SetData('Sensor1-AccX',Sensor1AccX)
    2. if all data is registed, execute Compile method
        e.g window.Compile(windowWidth=16,overlap=0.5)
       The data return by this method is the needed window
"""

import numpy as np


class Window:
    """Input dictionary
       Ouput Windows
    """         
    def __init__(self):                  
        self.seed ={}
        self.window=np.array([])
    
    # Regist Data to seed
    def SetData(self, RegistDataName, RegistData):
        if RegistDataName in self.seed:
            print RegistDataName+' has been registed before'
            return -1
        
        self.seed[RegistDataName] = RegistData
        print RegistDataName+' is registed now'
    
    # e.g 
    # seed has AccX,AccY,AccZ s data
    # array = [ArrayAccX,
    #          ArrayAccY,
    #          ArrayAccZ]
    # array.T =[ArrayAccX, ArrayAccY, ArrayAccZ]
    def _Build(self):
        keys = self.seed.keys()
        array = self.seed[keys[0]]
        
        for k in keys[1:]:
            array = np.vstack( (array,self.seed[k]))
        print 'Build Complete'
        return array.T
    
    def Compile(self,windowWidth,overlapNum):
        source = self._Build()
        # e.g source=[ArrayAccX, ArrayAccY, ArrayAccZ]
        # source.shape = [ len(seed[Acc*], 3]
        print source.shape 
       
        sourceColum  = source.ndim
        sourceRows= source.size
        print "sourceRows ="+str( sourceRows )

        windowNum=( (sourceRows-windowWidth)/overlapNum)
        print "windowData's num ="+str(windowNum)
        offset = sourceRows - (windowNum*overlapNum+windowWidth)
        print "SourceData's aborting data = "+str(offset)
        #offset = len(source)%windowWidth
        
        if offset % 2 == 1:
            offset=offset/2 +1
        else:
            offset=offset/2
            
        # first array
        start = offset
        goal = start+windowWidth
        self.window = source[start:goal].T
        
        for i in range(1,windowNum+1):
            start = start+overlapNum
            goal = start+windowWidth
            adding = source[start:goal].T
            self.window = np.dstack((self.window,adding))
        print "window shape is "+str( self.window.T.shape )
        return self.window.T

    def MakeWindowSet(self,windowWidth, overlapNum):
        window = self.Compile(windowWidth, overlapNum)
        d2window = self.d2window(wind=self.window)
        