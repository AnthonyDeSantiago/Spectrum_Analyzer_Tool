import os
import easyocr 
import cv2
from matplotlib import pyplot as plt
import numpy as np


class util():
    @staticmethod
    def getOCR(img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)
        result = reader.readtext(img, width_ths= 1)
        for r in result:
            print(r)
            bbox, text, score = r
            cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,0,), 2)
        
        plt.imshow(img)
        plt.show()

    @staticmethod
    def getReferenceLevel(img):
        img = cv2.imread(img)
        img = img[140:180, 195:290]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)
        result = reader.readtext(img, width_ths= 1)
        for r in result:
            print(r)
            bbox, text, score = r
            cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,0,), 2)
        
        plt.imshow(img)
        plt.show()

    @staticmethod
    def getSpan(img):
        img = cv2.imread(img)
        img = img[750:820, 700:900]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)
        result = reader.readtext(img, width_ths= 1)
        for r in result:
            print(r)
            bbox, text, score = r
            cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,0,), 2)
        
        plt.imshow(img)
        plt.show()

    @staticmethod
    def getCenter(img):
        img = cv2.imread(img)
        img = img[760:820, 270:470]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)
        result = reader.readtext(img, width_ths= 1)
        for r in result:
            print(r)
            bbox, text, score = r
            cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
            cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,0,), 2)
        
        plt.imshow(img)
        plt.show()



