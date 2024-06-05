# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:06:48 2020

@author: christine
"""

#from lib.clahe import CLAHE
#import numpy as np
#import cv2

# CLAHE
#img = cv2.imread( "images/clahe/2.jpg", 0 ).astype('uint8')

#for win in [2,5,20,50]:
    #cv2.imwrite( "result_"+str(win)+".png", CLAHE( img, win, win, 128, 5 ) )

# Contrast stretch
#img = cv2.imread( "nuclei.png", 0 ).astype('uint8')

#img = img.astype('float')
#img -= img.min()
#img /= img.max()
#img *= 255

#cv2.imwrite( "results/clahe/2.jpg", img )

import numpy as np
import cv2
bgr = cv2.imread('E:/yolo BTSD/images/clahe/2a.jpg')
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imwrite('E:/yolo BTSD/results/clahe/2aoutput.jpg',bgr)