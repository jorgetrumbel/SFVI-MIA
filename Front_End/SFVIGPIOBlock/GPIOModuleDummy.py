try:
    import RPi.GPIO as GPIO
    import gpiozero as GPIOZ
except:
    print("Error importing GPIO, not running on RPi")

import time, threading

pinNames = ("Trigger Pin", "Online Pin", "Busy Pin", "OK Pin", "NOK Pin")
PIN_TRIGGER_NAME = pinNames[0]
PIN_ONLINE_NAME = pinNames[1]
PIN_BUSY_NAME = pinNames[2]
PIN_OK_NAME = pinNames[3]
PIN_NOK_NAME = pinNames[4]

availablePins = (4,17,27,22,23)

pinDict = {availablePins[0]: "BOARD7",
           availablePins[1]: "BOARD11",
           availablePins[2]: "BOARD13",
           availablePins[3]: "BOARD15",
           availablePins[4]: "BOARD16"}

class IO():
    def __init__(self):
        self.pinStartup()
    
    def pinStartup(self):        
        pass
    
    def setTriggerPin(self, pin):
        pass
        
    def setOnlinePin(self, pin):
        pass
        
    def setBusyPin(self, pin):
        pass
        
    def setOKPin(self, pin):
        pass
        
    def setNOKPin(self, pin):
        pass
        
    def setTriggerPinFunc(self, func):
        pass
        
    
    def OnlinePinFunc(self, state):
        pass
    
    def setBusyPinFunc(self, state):
        pass
    
    def OKPinFunc(self):
        pass

    def NOKPinFunc(self):
        pass
    
    def triggerCallbackfunc(self):
        pass

    def triggerDebounceFunc(self):
        pass
    
    def closeAll(self):
        pass
    
    triggerPin = None
    triggerPinBtn = None
    triggerBtnDebounce = False
    triggerCallback = None
    onlinePin = None
    onlinePinLed = None
    busyPin = None
    busyPinLed = None
    OKPin = None
    OKPinLed = None
    NOKPin = None
    NOKPinLed = None


'''
#TEST PROGRAM
def myCB(io:IO):
    print("Callback")
    io.OKPinFunc()
    io.NOKPinFunc()
    

boardIO = IO()
boardIO.setTriggerPinFunc(lambda: myCB(boardIO))
boardIO.closeAll()

while(1):
    pass
'''