# server.py 
import socket                                         
import time
import os

# create a socket object
serversocket = socket.socket(
	        socket.AF_INET, socket.SOCK_STREAM) 

# get local machine name
host = socket.gethostname()                           

port = 9999                                           

# bind to the port
serversocket.bind((host, port))                                  

# queue up to 5 requests
serversocket.listen(5)                                           

while True:
    # establish a connection
    clientsocket,addr = serversocket.accept()      

    tm = clientsocket.recv(1024)                       


    print("Got a connection from %s" % str(host))
    currentTime = time.ctime(time.time()) + "\r\n"

    result = tm.split(':')[1]

    # print 'recived form server'+ tm

    print ' resutl clear' + result

    clientsocket.send(result)    
                   

    clientsocket.close()

