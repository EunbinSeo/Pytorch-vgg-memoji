import pyautogui
import os
import cv2
import time
import numpy as np

def grabImage(grabRegion, fileName):
    image = pyautogui.screenshot(region=grabRegion)
    image = cv2.resize((224,224))
    img_file = cv2.imread(fileName)

def process(img):
    #img = img*255
    mean = [129.1863,104.7624,93.5940]
    imageBgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float)
    imageBgr = np.transpose(imageBgr, (2, 1, 0))
    for i in range(len(mean)):
        imageBgr[:, :, i] -= mean[i]

    return imageBgr


def indexsortTable(tbl):
    idx = np.argsort(tbl)[::-1]
    return idx


def indexsort(tbl):
    return(indexsortTable(tbl)[0])

def sleep(n):
    time.sleep(n)

def readline():
    #os.execute("read") ????
    pass