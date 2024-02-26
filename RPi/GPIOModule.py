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
        #Configure Pins defaults
        self.setTriggerPin(availablePins[0])
        self.setOnlinePin(availablePins[1])
        self.setBusyPin(availablePins[2])
        self.setOKPin(availablePins[3])
        self.setNOKPin(availablePins[4])
        pass
    
    def setTriggerPin(self, pin):
        prevPin = self.triggerPin
        self.triggerPin = pin
        if prevPin:
            self.triggerPinBtn.close()
        #Configure pin
        self.triggerPinBtn = GPIOZ.Button(pinDict[pin], pull_up=True)
        self.triggerPinBtn.when_pressed = self.triggerCallbackfunc
        
    def setOnlinePin(self, pin):
        prevPin = self.onlinePin
        self.onlinePin = pin
        if prevPin:
            self.onlinePinLed.close()
        #Configure pin
        self.onlinePinLed = GPIOZ.LED(pinDict[pin])
        
    def setBusyPin(self, pin):
        prevPin = self.busyPin
        self.busyPin = pin
        if prevPin:
            self.busyPinLed.close()
        #Configure pin
        self.busyPinLed = GPIOZ.LED(pinDict[pin])
        
    def setOKPin(self, pin):
        prevPin = self.OKPin
        self.OKPin = pin
        if prevPin:
            self.OKPinLed.close()
        #Configure pin
        self.OKPinLed = GPIOZ.LED(pinDict[pin])
        
    def setNOKPin(self, pin):
        prevPin = self.NOKPin
        self.NOKPin = pin
        if prevPin:
            self.NOKPinLed.close()
        #Configure pin
        self.NOKPinLed = GPIOZ.LED(pinDict[pin])
        
    def setTriggerPinFunc(self, func):
        self.triggerCallback = func
        
    
    def OnlinePinFunc(self, state):
        if onlinePin:
            if state == True:
                self.onlinePinLed.on()
            elif state == False:
                self.onlinePinLed.off()
    
    def setBusyPinFunc(self, state):
        if self.busyPin:
            if state == True:
                self.busyPinLed.on()
            elif state == False:
                self.busyPinLed.off()
    
    def OKPinFunc(self):
        if not self.OKPinLed.is_lit:
            self.OKPinLed.blink(on_time = 0.5, n = 1)

    def NOKPinFunc(self):
        if not self.NOKPinLed.is_lit:
            self.NOKPinLed.blink(on_time = 0.5, n = 1)
    
    def triggerCallbackfunc(self):
        if self.triggerBtnDebounce == False:
            self.triggerCallback()
            print("triggered") #FOR DEBUGGING
            self.triggerBtnDebounce = True
            threading.Timer(1, self.triggerDebounceFunc).start()

    def triggerDebounceFunc(self):
        self.triggerBtnDebounce = False
    
    def closeAll(self):
        self.triggerPinBtn.close()
        self.onlinePinLed.close()
        self.busyPinLed.close()
        self.OKPinLed.close()
        self.NOKPinLed.close()
    
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