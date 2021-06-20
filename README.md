# Pro_Covid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
#
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from os.path import dirname, join
import numpy as np
import imutils
import time
import cv2
import os
#IMPORTACION DEL SENSOR
from smbus2 import SMBus
from mlx90614 import MLX90614
import TM1638
import drivers
import time
import sys

from time import sleep
import datetime

import RPi.GPIO as GPIO
display = drivers.Lcd()

DIO = 16
CLK = 20
STB = 21
dis = TM1638.TM1638(DIO, CLK, STB)
dis.enable(1)
def detect_and_predict_mask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY)
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)
def Foto():
    captura = cv2.VideoCapture(0)
    salida = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
    while (captura.isOpened()):
        ret, imagen = captura.read()
        if ret == True:
            cv2.imshow('video', imagen)
            salida.write(imagen)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else: break
    captura.release()
    salida.release()
    cv2.destroyAllWindows()

def TM1638():
    for i in range(8):
        for j in range(7):
            m = 128 >> j;
            dis.send_char(i-1, m)
            time.sleep(0.02)
    count = 0
    while True:
        dis.set_text(str(count))
        count += 100
        time.sleep(0.02)
        if GPIO.input(SENSOR_PIR)==1:
            break

def Admitir():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(5, GPIO.OUT)
    GPIO.output(5, True)    
    sleep(3)
    GPIO.output(5, False)
    sleep(0.5)

def noAdmitir():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(13, GPIO.OUT)
    #while True:
    GPIO.output(13, True)   
    sleep(3)
    GPIO.output(13, False)
    sleep(0.5)

def inicio():
    display.lcd_display_string("HOLA!!!", 1)
    sleep(1)
    display.lcd_display_string("...BIENVENIDO...", 2)
    sleep(2)
    display.lcd_clear()

prototxtPath = r"/home/pi/esete no.3/lcd/face_detector/deploy.prototxt"
weightsPath = r"/home/pi/esete no.3/lcd/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

inicio()
path = os.getcwd()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
SENSOR_PIR = 24
GPIO.setup(SENSOR_PIR,GPIO.IN)
print ("Sensor estabilizandonse")
time.sleep(2)   
print("Sistema Activado")
print("Iniciando Video..")
vs = VideoStream(src=0).start()

try:
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if GPIO.input(SENSOR_PIR)==1:
                print("Persona Detectada")
                bus = SMBus(1)
                sensor = MLX90614(bus, address=0x5A)
                print("Temperatura Ambiente : ","{:.2f}".format(sensor.get_ambient()))
                print("Temperatura Coorporal:", "{:.2f}".format(sensor.get_object_1()))
                temp = sensor.get_object_1()
                result = str(temp)+" C"
                if temp<37 and mask > withoutMask:
                    time.sleep(0.1)
                    display.__init__()
                    display.lcd_display_string("Listo!!!", 1)
                    display.lcd_display_string("Temp: "+ result, 2)
                    sleep(5)
                    display.lcd_clear()
                    Admitir()
                    label = "Gracias.Con mascarilla"
                    color = (0, 255, 0)
                else:
                    time.sleep(0.01)
                    label = "Mascarilla NO Detectada"
                    color = (0, 0, 255)
                    display.lcd_display_string("Alto!!!!...", 1)
                    display.lcd_display_string("Temp: "+ result, 2)
                    sleep(5)
                    display.lcd_clear()
                    noAdmitir()
                    Foto()
                    Buzzer()
                    enviarM()
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)  
                bus.close()
                GPIO.input(SENSOR_PIR)==0
                print("Esperando Persona...")
                TM1638()
                time.sleep(0.5)
                  
        cv2.imshow("Detector de Mascarillas", frame)
        key = cv2.waitKey(1) & 0xFF

