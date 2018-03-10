
# -*- coding: utf-8 -*-
# Copyright (c) 2017, Frappe Technologies and contributors
# For license information, please see license.txt

from __future__ import unicode_literals
import frappe
from frappe.model.document import Document
from websocket_server import WebsocketServer
import json
import logging
import datetime

def mad():
    import socket                                         
    import time
    import os

    # create a socket object
    serversocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM) 

    # get local machine name
    host2 = socket.gethostname()                           

    port3 = 9999                                           

    # bind to the port
    serversocket.bind((host2, port3))                                  

    # queue up to 5 requests
    serversocket.listen(5)   

    print "listen to 9999"                                       

    while True:
        # establish a connection
        clientsocket,addr = serversocket.accept()      

        tm = clientsocket.recv(1024)                       


        print("Got a connection from %s" % str(host2))
        currentTime = time.ctime(time.time()) + "\r\n"

        result = tm.split(':')[1]

        # print 'recived form server'+ tm

        print ' resutl clear' + result


        # clientsocket.close()

        return result                   

# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    # server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):

    print("################# Start Time ####################")
    print(datetime.datetime.now())
    if len(message) > 200:
        message = message[:200]+'..'
    # print("Client(%d) said: %s" % (client['id'], message))
    print("Get Serial Number of User's Device")
    print("Compare with the Serial Number stored in database")
    print("Get Serial Number of User Device")
    max = json_mylist = json.dumps(["result",mad()])

    server.send_message_to_all(max)

    print(datetime.datetime.now())
    print("################# End Time ####################")


PORT=9001
server = WebsocketServer(PORT)
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()



