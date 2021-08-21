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
    mean = [129.1863,104.7624,93.5940]
    img = cv2.resize(img,(224,224))
    imageBgr = np.transpose(img, (2, 1, 0)).astype(np.float)
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