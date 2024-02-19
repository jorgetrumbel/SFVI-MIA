import sys
from io import StringIO

class Log(object):
    def __init__(self):
        self.orgstdout = sys.stdout
        #self.log = open("log.txt", "a")
        self.log = sys.stdout

    def write(self, msg):
        self.orgstdout.write(msg)
        self.log.write(msg)  

    def flush(self):
        sys.__stdout__.flush()

    encoding = sys.__stdout__.encoding


sys.stdout = Log()
sys.stdout.log = buffer = StringIO()