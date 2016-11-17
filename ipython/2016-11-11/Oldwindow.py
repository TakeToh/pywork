"""
    how to use class Window
    1. Regist data using SetData method
        e.g window.SetData('Sensor1-AccX',Sensor1AccX)
    2. if all data is registed, execute Compile method
        e.g window.Compile(windowWidth=16,overlap=0.5)
       The data return by this method is the needed window
"""

import numpy as np
# 表示用
from tqdm import tqdm

class Window:
    """Input dictionary
       Ouput Windows
    """         
    def __init__(self):                  
        self.seed ={}
        self.window=np.array([])
    
    def SetData(self, RegistDataName, RegistData):
        if RegistDataName in self.seed:
            print RegistDataName+' has been registed before'
            return -1
        
        self.seed[RegistDataName] = RegistData
        print RegistDataName+' is registed now'
    
    def _Build(self):
        keys = self.seed.keys()
        array = self.seed[keys[0]]
        
        for k in keys[1:]:
            array = np.vstack( (array,self.seed[k]))
        print 'Build Complete'
        return array.T
    
    def Compile(self,windowWidth,slidingWidth):
        source = self._Build()
        print 'Source shape is'+str( source.shape )
       
        # Kind of Data
        numData = source.ndim
        # Length of Data
        lengthData = source.size
        
        # Num WindowFrame
        numWindow = (lengthData-windowWidth)/(2*slidingWidth)
        print 'window frames num ='+str(numWindow)
        
        offset = lengthData-(windowWidth+slidingWidth*numWindow)
        print 'offset='+str(offset)
        
        if offset % 2 == 1:
            offset=offset/2 +1
        else:
            offset=offset/2
            
        # first array
        start = offset
        goal = start+windowWidth
        self.window = source[start:goal].T
        
        for i in tqdm( range(1,len(source)/windowWidth) ):
            start = start+slidingWidth
            goal = start+windowWidth
            
            nextFrame = source[start:goal].T
            
            self.window = np.dstack((self.window,nextFrame))
        return self.window.T

    def MakeWindowSet(self,windowWidth, overlap):
        window = self.Compile(windowWidth, overlap)
        d2window = self.d2window(wind=self.window)
        