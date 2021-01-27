import os
import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
import json
import datetime
import numpy as np
import skimage.draw
import cv2
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils 

import matplotlib.pyplot as plt

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "TraiCay"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 515 # 77 images

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9 

def PhatHienNhanDang(model, image, imagecv):
        class_names = ['BG','Buoi','Cam','Coc', 'Khe', 'Mit']
        r = model.detect([image], verbose=1)[0]
        print(r['masks'])
        print(r['rois'])
        print(r['scores'])
        print(r['class_ids'])
        mask = r['masks']
        L = len(r['rois'])
        for i in range(0, L):
            y1 = r['rois'][i][0]
            x1 = r['rois'][i][1]
            y2 = r['rois'][i][2]
            x2 = r['rois'][i][3]
            cv2.rectangle(imagecv,(x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(imagecv,class_names[r['class_ids'][i]],(x1+5, y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        #visMask = (mask * 255).astype("uint8")
        #contours, _ = cv2.findContours(visMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #for i in range(len(contours)):
        #    cv2.drawContours(imagecv, contours, i, (0,255,0), 2)
        cv2.imshow('ImageIn',imagecv)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()


def onOpen():
    global ftypes
    ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
    dlg = Open(filetypes = ftypes)
    fl = dlg.show()
  
    if fl != '':
        global imgin
        global imgincv
        imgin = skimage.io.imread(fl)
        #imgin = cv2.imread(fl,cv2.IMREAD_COLOR);
        cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
        imgincv = cv2.cvtColor(imgin, cv2.COLOR_BGR2RGB)      
        cv2.imshow("ImageIn", imgincv)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

def onNhanDangTraiCay():
    PhatHienNhanDang(model, imgin, imgincv)

root = Tk()
menubar = Menu(root)
root.config(menu=menubar)
root.title("Nhan dang Trai cay")

fileMenu = Menu(menubar)
fileMenu.add_command(label="Open", command=onOpen)
fileMenu.add_command(label="Nhan dang Trai cay", command=onNhanDangTraiCay)
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=quit)
menubar.add_cascade(label="File", menu=fileMenu)


if __name__ == '__main__':
    class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display() 

    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir='logs')
    weights_path = 'mask_rcnn_traicay_0029.h5'
    model.load_weights(weights_path, by_name=True)
    root.geometry("300x285+0+100")
    root.mainloop()



