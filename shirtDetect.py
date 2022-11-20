import cv2
import numpy as np
import os
import time
import imutils
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import threading

# from color_detection import color_detection
def get_dress(file):
    global detecting, stop, output_img, x, y, w, h
    file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
    rgb  = file.numpy()
    file = np.expand_dims(file,axis=0)/ 255.
    seq = pretrained_model.predict(file, verbose = 0)
    seq = seq[3][0,:,:,0]
    seq = np.expand_dims(seq,axis=-1)
    dummy = np.ones((rgb.shape[0],rgb.shape[1],1))
    rgbx = np.concatenate((rgb,dummy*255),axis=-1)

    frame = seq*255
    frame = frame.astype(np.uint8)
    # Convert the grayscale image to binary
    _, binary = cv2.threshold(frame, 100, 255, cv2.THRESH_OTSU)
    
    # Find the contours on the binary image, and store them in a list
    # Contours are drawn around white blobs.
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw a bounding box around all contours
    rgbx = rgbx.astype(np.uint8)
    w = 0   
    for c in contours:
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 10000:
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(rgbx,(x,y), (x+w,y+h), (255,0,0), 3)

    detecting = False
    stop = True
    # return rgbx/255, False
    return

pretrained_model = load_model("save_ckp_frozen.h5")
detecting = False
stop = False
x, y, w, h = 0, 0, 0, 0

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    # gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    # cv2.imshow('Video', frames)
    file = tf.image.resize_with_pad(frames,target_height=512,target_width=512)
    rgb  = file.numpy()
    dummy = np.ones((rgb.shape[0],rgb.shape[1],1))
    rgbx = np.concatenate((rgb,dummy*255),axis=-1)
    output_img = rgbx/255

    # Detecting Shirt
    if detecting == False and stop == False:
        detecting = True
        xt = threading.Thread(target=get_dress, args=(frames,))
        xt.start()
    # Remove thread
    if stop == True:
        xt.join()
        stop = False

    # Draw shirt area
    if w != 0:
        cv2.rectangle(output_img,(x,y), (x+w,y+h), (255,0,0), 3)
    cv2.imshow('Shirt', output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
