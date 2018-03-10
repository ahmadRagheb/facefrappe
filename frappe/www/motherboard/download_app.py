# client.py  
import socket
import os
import time

# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# get local machine name
host = socket.gethostname()                           

port = 9999

# connection to hostname on the port.
s.connect((host, port))                               


# Receive no more than 1024 bytes
tm = s.recv(1024)                       


print("Got a connection from %s" % str(host))
currentTime = time.ctime(time.time()) + "\r\n"


print 'recived form server'+ tm


s.close()
