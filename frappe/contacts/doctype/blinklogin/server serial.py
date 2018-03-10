
# -*- coding: utf-8 -*-
# Copyright (c) 2017, Frappe Technologies and contributors
# For license information, please see license.txt




from __future__ import unicode_literals
import frappe
from frappe.model.document import Document

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition
import base64

import json
import logging
from websocket_server import WebsocketServer

from PIL import Image

import io

from io import BytesIO
from io import StringIO

from StringIO import StringIO
import urllib2
import socket
import time
import os

clients = {}


one_user_img=''
numer_of_faces_in_img=0


def fankosh(request):

    print("Got a connection from %s" % str(host))
    currentTime = time.ctime(time.time()) + "\r\n"

    result = tm.split(':')[1]

    # print 'recived form server'+ tm

    print ' resutl clear' + result
    return result


def client_left(client, server):
    # msg = "Client (%s) left" % client['id']
    # print msg
    try:
        clients.pop(client['id'])
    except:
        print "Error in removing client %s" % client['id']
    for cl in clients.values():
        server.send_message(cl, msg)


def new_client(client, server):
    msg = "New client (%s) connected" % client['id']
    print msg
    for cl in clients.values():
        server.send_message(cl, msg)
    clients[client['id']] = client


def msg_received(client, server, msg):

    clientid = client['id']

    for cl in clients:
            if cl == clientid:
                processed_image = fankosh(msg)
                json_mylist = json.dumps(["serial",processed_image])
                server.send_message(cl, json_mylist)




server = WebsocketServer(9001,host='0.0.0.0')
# server = WebsocketServer(9001,host=str(socket.gethostname()))
server.set_fn_client_left(client_left)
server.set_fn_new_client(new_client)
"""Sets a callback function that will be called when a client sends a message """
# server.set_fn_message_received(msg_received)
server.run_forever()



