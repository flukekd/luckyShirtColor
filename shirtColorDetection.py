import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import imutils

class gd(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.new_im = None
        
    def run(self):
        global frames
        while not self.stop_event.is_set():
            self.find_shirt()
            self.cropped_shirt()
            if self.new_im is not None:
                self.colour_detection()
            

    def find_shirt(self):
        self.file = tf.image.resize_with_pad(frames,target_height=512,target_width=512)
        self.rgb  = self.file.numpy()
        self.file = np.expand_dims(self.file,axis=0)/ 255.
        self.seq = pretrained_model.predict(self.file, verbose = 0)
        self.seq = self.seq[3][0,:,:,0]
        self.seq = np.expand_dims(self.seq, axis=-1)
        self.seq[self.seq < 0.95] = 0
        self.seq[self.seq >= 0.95] = 1

        self.c1x = self.rgb * self.seq
        self.c2x = self.rgb * (1-self.seq)
        self.cfx = self.c1x + self.c2x

        self.dummy = np.ones((self.rgb.shape[0], self.rgb.shape[1], 1))
        self.rgbx = np.concatenate((rgb, self.dummy*255), axis=-1)

        self.frame = self.seq*255
        self.frame = self.frame.astype(np.uint8)

    def cropped_shirt(self):
        # Convert the grayscale image to binary
        _, self.binary = cv2.threshold(self.frame, 100, 255, cv2.THRESH_OTSU)
        
        # Find the contours on the binary image, and store them in a list
        # Contours are drawn around white blobs.
        self.contours, _ = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw a bounding box around all contours
        self.rgbx = self.rgbx.astype(np.uint8)

        self.w = 0
        
        for c in self.contours:
            # Make sure contour area is large enough
            if (cv2.contourArea(c)) > 10000:
                self.x, self.y, self.w, self.h = cv2.boundingRect(c)
                # cv2.rectangle(rgbx,(x,y), (x+w,y+h), (255,0,0), 3)
                # create new image of desired size and color (blue) for padding
                desired_size = 500
                im_pth = frames
                im = im_pth
                old_size = im.shape[:2] # old_size is in (height, width) format

                ratio = float(desired_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])

                # new_size should be in (width, height) format
                im = cv2.resize(im, (new_size[1], new_size[0]))

                delta_w = desired_size - new_size[1]
                delta_h = desired_size - new_size[0]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)

                color = [0, 0, 0]
                self.new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                self.new_im = self.new_im[thread.y:thread.y+thread.h, thread.x:thread.x+thread.w] 

    def colour_detection(self):
        hsv = cv2.cvtColor(self.new_im, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])

        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])

        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 200, 200])

        lower_pink = np.array([150, 50, 50])
        upper_pink = np.array([170, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 30, 255])

        mask1 = cv2. inRange (hsv, lower_yellow,upper_yellow)
        mask2 = cv2. inRange (hsv, lower_green,upper_green)
        mask3 = cv2. inRange (hsv, lower_red,upper_red)
        mask4 = cv2. inRange (hsv, lower_blue,upper_blue)
        mask5 = cv2. inRange (hsv, lower_cyan,upper_cyan)
        mask6 = cv2. inRange (hsv, lower_purple,upper_purple)
        mask7 = cv2. inRange (hsv, lower_brown,upper_brown)
        mask8 = cv2. inRange (hsv, lower_pink,upper_pink)
        mask9 = cv2. inRange (hsv, lower_black,upper_black)

        cnts1 = cv2. findContours (mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)
        cnts1 = sorted(cnts1, key=cv2.contourArea, reverse=True)
        cnts1 = cnts1[:3]

        cnts2 = cv2. findContours (mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)
        cnts2 = cnts2[:3]

        cnts3 = cv2. findContours (mask3, cv2 .RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts3 = imutils.grab_contours(cnts3)
        cnts3 = sorted(cnts3, key=cv2.contourArea, reverse=True)
        cnts3 = cnts3[:3]

        cnts4 = cv2.findContours (mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts4 = imutils.grab_contours(cnts4)
        cnts4 = sorted(cnts4, key=cv2.contourArea, reverse=True)
        cnts4 = cnts4[:3]

        cnts5 = cv2. findContours (mask5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts5 = imutils.grab_contours(cnts5)
        cnts5 = sorted(cnts5, key=cv2.contourArea, reverse=True)
        cnts5 = cnts5[:3]

        cnts6 = cv2. findContours (mask6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts6 = imutils.grab_contours(cnts6)
        cnts6 = sorted(cnts6, key=cv2.contourArea, reverse=True)
        cnts6 = cnts6[:3]

        cnts7 = cv2. findContours (mask7, cv2 .RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts7 = imutils.grab_contours(cnts7)
        cnts7 = sorted(cnts7, key=cv2.contourArea, reverse=True)
        cnts7 = cnts7[:3]

        cnts8 = cv2.findContours (mask8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts8 = imutils.grab_contours(cnts8)
        cnts8 = sorted(cnts8, key=cv2.contourArea, reverse=True)
        cnts8 = cnts8[:3]

        cnts9 = cv2.findContours (mask9, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts9 = imutils.grab_contours(cnts9)
        cnts9 = sorted(cnts9, key=cv2.contourArea, reverse=True)
        cnts9 = cnts9[:3]

        for c in cnts1:
            area1 = cv2.contourArea(c)
            if area1 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments (c)

                cx = int(M["m10"]/ M["m00"])
                cy = int(M["m01"]/M["m00"])

                # self.label = ["Yellow"]
                print('Yellow')
                # cv2.circle(self.new_im,(cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Yellow", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)

        for c in cnts2:

            area2 = cv2.contourArea(c)
            if area2 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments (c)

                cx = int(M["m10"]/M["m00"])

                cy = int(M["m01"]/M["m00"])
                print('Green')
                # cv2.circle(self.new_im,(cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Green", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)


        for c in cnts3:

            area3 = cv2.contourArea(c)
            if area3 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments(c)

                cx = int(M["m10"]/ M["m00"])

                cy = int (M["m01"]/M["m00"])
                print('red')
                # cv2.circle(self.new_im, (cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "red", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)

        for c in cnts4:

            area4 = cv2.contourArea(c)
            
            if area4 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)


                M = cv2.moments (c)


                cx = int (M["m10"]/ M["m00"])
                cy = int (M["m01"]/M["m00"])
                print('blue')
                # cv2.circle(self.new_im, (cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "blue", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)


        for c in cnts5:

            area5 = cv2.contourArea(c)
            if area5 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments (c)

                cx = int(M["m10"]/ M["m00"])
                cy = int(M["m01"]/M["m00"])
                print('cyan')
                # cv2.circle(self.new_im,(cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Cyan", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)

        for c in cnts6:

            area6 = cv2.contourArea(c)
            if area6 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments (c)

                cx = int(M["m10"]/M["m00"])

                cy = int(M["m01"]/M["m00"])
                print('purp')
                # cv2.circle(self.new_im,(cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Purple", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)


        for c in cnts7:

            area7 = cv2.contourArea(c)
            if area7 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)

                M = cv2.moments(c)

                cx = int(M["m10"]/ M["m00"])

                cy = int (M["m01"]/M["m00"])
                print('brown')
                # cv2.circle(self.new_im, (cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Brown", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)

        for c in cnts8:

            area8 = cv2.contourArea(c)
            
            if area8 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)


                M = cv2.moments (c)


                cx = int (M["m10"]/ M["m00"])
                cy = int (M["m01"]/M["m00"])
                print('pink')
                # cv2.circle(self.new_im, (cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Pink", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)

        for c in cnts9:

            area9 = cv2.contourArea(c)
            # print(area9)
            if area9 > 5000:

                cv2.drawContours(self.new_im,[c],-1,(0,255,0),3)


                M = cv2.moments (c)


                cx = int (M["m10"]/ M["m00"])
                cy = int (M["m01"]/M["m00"])
                print('black')
                # cv2.circle(self.new_im, (cx,cy),7,(255,255,255),-1)
                # cv2.putText(self.new_im, "Black", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5, (255,255,255), 3)
                
    def stop(self):
        self.stop_event.set()

video_capture = cv2.VideoCapture(0)
pretrained_model = load_model("save_ckp_frozen.h5")
ret, frames = video_capture.read()
thread = gd()
thread.start()



while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    
    # Show input image in the dimension of 512 x 512
    file = tf.image.resize_with_pad(frames,target_height=512,target_width=512)
    rgb  = file.numpy()
    dummy = np.ones((rgb.shape[0],rgb.shape[1],1))
    rgbx = np.concatenate((rgb,dummy*255),axis=-1)
    output_img = rgbx/255

    # Draw shirt area
    if thread.w != 0:
        cv2.rectangle(output_img,(thread.x,thread.y), (thread.x+thread.w,thread.y+thread.h), (255,0,0), 3)
    cv2.imshow('Shirt', output_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        thread.stop()
        thread.join()   
        break
video_capture.release()
cv2.destroyAllWindows()
