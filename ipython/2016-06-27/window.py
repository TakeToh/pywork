"""
    how to use class Window
    1. Regist data using SetData method
        e.g window.SetData('Sensor1-AccX',Sensor1AccX)
    2. if all data is registed, execute Compile method
        e.g window.Compile(windowWidth=16,overlap=0.5)
       The data return by this method is the needed window
"""

<<<<<<< HEAD
import numpy as np


=======
>>>>>>> origin/master
class Window:
    """Input dictionary
       Ouput Windows
    """         
    def __init__(self):                  
        self.seed ={}
<<<<<<< HEAD
        self.window=np.array([])
=======
>>>>>>> origin/master
    
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
    
    def Compile(self,windowWidth,overlap):
        source = self._Build()
        print source.shape
       
        sourceRows = source.ndim
        sourceColum = source.size
        offset = len(source)%windowWidth
        if offset % 2 == 1:
            offset=offset/2 +1
        else:
            offset=offset/2
        # first array
        start = offset
        goal = start+windowWidth
<<<<<<< HEAD
        self.window = source[start:goal].T
=======
        window = source[start:goal].T
>>>>>>> origin/master
        
        for i in range(1,len(source)/windowWidth):
            start = start+ int(windowWidth*overlap)
            goal = start+windowWidth
            adding = source[start:goal].T
<<<<<<< HEAD
            self.window = np.dstack((self.window,adding))
        return self.window.T
    
    def d2window(self,wind=self.window):
        output = wind.reshape((len(wind),np.prod(wind.shape[1:])))
        return output
    
    def MakeWindowSet(self,windowWidth, overlap):
        window = self.Compile(windowWidth, overlap)
        d2window = self.d2window(wind=self.window)
        
=======
            window = np.dstack((window,adding))
        return window.T
>>>>>>> origin/master
