# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:17:20 2020

@author: USER
"""
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

path = os.listdir(args["image"])

def get_color(image):
    #加载原图
    img=cv2.imread(image)
    print('img:',type(img),img.shape,img.dtype)

    # 轉換圖片色彩RBG --> HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # print image
    # cv2.imshow('hsv',hsv)

    #提取蓝色区域
    # Red color
    low_red1 = np.array([155, 25, 30])
    high_red1 = np.array([179, 255, 255])
    mask1=cv2.inRange(hsv,low_red1,high_red1)

    low_red2 = np.array([0, 25, 30])
    high_red2 = np.array([3, 255, 255])
    mask2=cv2.inRange(hsv,low_red2,high_red2)

    mask_r = cv2.bitwise_or(mask1, mask2)
#     print('mask_r',type(mask_r),mask_r.shape)
#     cv2.imshow('mask_r',mask_r)
    
    low_blue = np.array([105,50,50])
    high_blue = np.array([130,255,255])
    mask_b=cv2.inRange(hsv,low_blue,high_blue)
    
    mask = cv2.bitwise_or(mask_r, mask_b)
#     print('mask',type(mask),mask.shape)
#     cv2.imshow('mask',mask)

    #模糊
    blurred=cv2.blur(mask,(2,2))
    # cv2.imshow('blurred',blurred)

    #二值化
    ret,binary=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('blurred binary',binary)

    #使区域闭合无空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed',closed)

    #腐蚀和膨胀
    '''
    腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    '''
    erode=cv2.erode(closed,None,iterations=1)
    # cv2.imshow('erode',erode)

    dilate=cv2.dilate(erode,None,iterations=1)
    # cv2.imshow('dilate',dilate)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return dilate, img

def cut_image(dilate, img):
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "BTSD.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    #weightsPath = os.path.sep.join([args["yolo"], "yolov3BTSD_best.weights"])
    #configPath = os.path.sep.join([args["yolo"], "yolov3BTSD.cfg"])
    
      # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3sppBTSD_best.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3sppBTSD.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # 查找轮廓
    contours, hierarchy=cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print('輪廓個數：',len(contours))
    i=0
    res=img.copy()
    print(contours)
    for con in contours:
        area = cv2.contourArea(con)
        if area < 500 or area > 80000:
            continue;
        print('area:' + str(area))
        
        # 轮廓转换为矩形
        rect=cv2.minAreaRect(con)
        # 矩形转换为box
        box=np.int0(cv2.boxPoints(rect))
        # 在原图画出目标区域
        # cv2.drawContours(res,[box],-1,(0,0,255),2)
        # print([box])
        # 计算矩形的行列
        h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
        l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
        # print('h1',h1)
        # print('h2',h2)
        # print('l1',l1)
        # print('l2',l2)
        if h2<0:h2=0
        if l2<0:l2=0
        # 加上防错处理，确保裁剪区域无异常
        if h1-h2>0 and l1-l2>0:
            # 裁剪矩形区域
            temp=img[h2:h1,l2:l1]
            (H, W) = temp.shape[:2]
            Size_Class = ""
            if H*W < 1024:
                Size_Class = "S"
            elif H*W > 1024 and H*W < 9216:
                Size_Class = "M"
            else:
                Size_Class = "L"
            dw = 1./W
            dh = 1./H
            X = (l1+ l2)/2.0
            y = (h1 + h2)/2.0
            W = l1 - l2
            h = h1 - h2
            X = X*dw
            W = W*dw
            y = y*dh
            h = h*dh

            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(temp, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            detect_time = end - start
            # show timing information on YOLO
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    print('img',img)
                    print(X,y)
                    print(W, h)
                    cv2.rectangle(img, (l1, h1), (l2 , h2), color, 2)
                    text = "{}: {:.4f}\nSize: {}\nTime: {:.4f}".format(LABELS[classIDs[i]], confidences[i], Size_Class, detect_time)
                    text1 = "{}; {:.4f};Size; {};Time; {:.4f}".format(LABELS[classIDs[i]], confidences[i], Size_Class, detect_time)

                    file_name = save_path + 'result.txt'

                    if os.path.exists(file_name):
                        if os.path.isfile(file_name):
                            f = open(file_name, 'a')
                    else:
                        f = open(file_name, 'w')
                    f.write(num + ';')
                    f.write(text1 + '\n')
                    f.close()

                    y0, dy = 50, 20
                    for j, line in enumerate(text.split('\n')):
                        y = y0 + j*dy
                        cv2.putText(img, line, (l1-10, h1 - y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # show the output image
            #cv2.imshow("Image", img)
            cv2.imwrite(save_path + num + '.jpg', img)
            #cv2.imwrite("test.jpg", img)
            #cv2.waitKey(0)
            i=i+1
            # 显示裁剪后的标志
            #cv2.imwrite("results/" + num + '/sign' + str(i) + '.jpg', temp)
            # cv2.imshow('sign'+str(i),temp)

    #显示画了标志的原图       
    #cv2.imshow('res',res)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import os
import shutil

images = os.listdir(args["image"])
# print(images[0][:-4])

# if len(os.listdir("results/")) != 0:
#     for i in os.listdir("results/"):
#         shutil.rmtree("results/"+i)
save_path = "output/01/yolov3spp2/"

for i in images:
#     os.mkdir("results/" + i[:-4])
     
     pic = args["image"] + i 
     num = i[:-4]
     dilate, img = get_color(pic)
     cut_image(dilate, img)
#dilate, img = get_color(args["image"])
#cut_image(dilate, img)