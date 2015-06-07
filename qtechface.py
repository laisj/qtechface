# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:56:45 2015

@author: lai
"""
import cv2
import os
import configobj
import logging
import time

if not os.path.isdir("./log/"):
    os.makedirs("./log/")
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='./log/face_detect.log',
                    filemode='w')
logging.info('hi, logger')

config = configobj.ConfigObj("config.ini")
inputfolder = config["Runtime"]["inputfolder"]
outputfolder = config["Runtime"]["outputfolder"]
debugfolder = config["Runtime"]["debugfolder"]
trainingdatafile = config["Runtime"]["trainingdatafile"]
maxinterval = float(config["Runtime"]["maxinterval"])
detectparam1 = float(config["TrainingParam"]["detect1"])
detectparam2 = int(config["TrainingParam"]["detect2"])


if not os.path.isdir(inputfolder):
    os.makedirs(inputfolder)
if not os.path.isdir(outputfolder):
    os.makedirs(outputfolder)
if not os.path.isdir(debugfolder):
    os.makedirs(debugfolder)
face_cascade = cv2.CascadeClassifier(trainingdatafile)
for dirname, dirnames, filenames in os.walk(inputfolder):
    logging.error(filenames)
    for filename in filenames:
        if filename.endswith('.jpg'):
            print "last modified: %s" % time.ctime(os.path.getmtime(inputfolder + filename))
            print "created: %s" % time.ctime(os.path.getctime(inputfolder + filename))
            print "now: %s" % time.ctime()
            if time.time() - os.path.getmtime(inputfolder + filename) > maxinterval:
                print time.time() - os.path.getmtime(inputfolder + filename)
                print "timeout, pass"
                continue
            img = cv2.imread(inputfolder + filename)
            faces = face_cascade.detectMultiScale(img, detectparam1, detectparam2)
            index = 0
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                tempface = img[y:y+h, x:x+w]
                facefilename = outputfolder + filename[0:-4] + '_' + str(index) + '.jpg'
                logging.info(facefilename)
                cv2.imwrite(facefilename, tempface)
                index += 1
#                cv2.imshow('img', tempface)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
            
            logging.warning(filename)
            cv2.imwrite(debugfolder + filename, img)
#            cv2.imshow('img', img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
