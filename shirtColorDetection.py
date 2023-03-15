import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import imutils
from datetime import datetime


class get_dress(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.new_im = None
        self.c = None
        self.cx = 0
        self.cy = 0

    def run(self):
        global frames
        while not self.stop_event.is_set():
            self.find_shirt()
            self.cropped_shirt()
            if self.new_im is not None:
                self.colour_detection()

    def find_shirt(self):
        self.file = tf.image.resize_with_pad(
            frames, target_height=512, target_width=512)
        self.rgb = self.file.numpy()
        self.file = np.expand_dims(self.file, axis=0) / 255.
        self.seq = pretrained_model.predict(self.file, verbose=0)
        self.seq = self.seq[3][0, :, :, 0]
        self.seq = np.expand_dims(self.seq, axis=-1)
        self.seq[self.seq < 0.95] = 0
        self.seq[self.seq >= 0.95] = 1

        self.c1x = self.rgb * self.seq
        self.c2x = self.rgb * (1 - self.seq)
        self.cfx = self.c1x + self.c2x

        self.dummy = np.ones((self.rgb.shape[0], self.rgb.shape[1], 1))
        self.rgbx = np.concatenate((rgb, self.dummy * 255), axis=-1)

        self.frame = self.seq * 255
        self.frame = self.frame.astype(np.uint8)

    def cropped_shirt(self):
        # Convert the grayscale image to binary
        _, self.binary = cv2.threshold(self.frame, 100, 255, cv2.THRESH_OTSU)

        # Find the contours on the binary image, and store them in a list
        # Contours are drawn around white blobs.
        self.contours, _ = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a bounding box around all contours
        self.rgbx = self.rgbx.astype(np.uint8)

        # Reset self boarder location
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
                # old_size is in (height, width) format
                old_size = im.shape[:2]

                ratio = float(desired_size) / max(old_size)
                new_size = tuple([int(x * ratio) for x in old_size])

                # new_size should be in (width, height) format
                im = cv2.resize(im, (new_size[1], new_size[0]))

                delta_w = desired_size - new_size[1]
                delta_h = desired_size - new_size[0]
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)

                color = [0, 0, 0]
                self.new_im = cv2.copyMakeBorder(
                    im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                self.new_im = self.new_im[thread.y:thread.y +
                                          thread.h, thread.x:thread.x + thread.w]

    def colour_detection(self):

        def find_contours(hsv, lower, upper):
            mask = cv2.inRange(hsv, lower, upper)
            cnts = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            return cnts[:1]

        hsv = cv2.cvtColor(self.new_im, cv2.COLOR_BGR2HSV)

        # Color name, lower, upper
        colors = [("Yellow", np.array([20, 100, 100]), np.array([30, 255, 255])),
                  ("Green", np.array([40, 40, 40]), np.array([70, 255, 255])),
                  ("Red", np.array([0, 50, 50]), np.array([10, 255, 255])),
                  ("Blue", np.array([110, 50, 50]), np.array([130, 255, 255])),
                  ("Black", np.array([0, 0, 100]), np.array([180, 255, 200])),
                  ("Cyan", np.array([80, 100, 100]),
                   np.array([100, 255, 255])),
                  ("Purple", np.array([130, 50, 50]),
                   np.array([160, 255, 255])),
                  ("Brown", np.array([10, 50, 50]), np.array([20, 200, 200])),
                  ("Pink", np.array([150, 50, 50]), np.array([170, 255, 255]))
                  ]

        cnts_list = [(color_name, find_contours(hsv, lower, upper))
                     for color_name, lower, upper in colors]

        area_min = 5000
        max_cnt = [None] * 4
        for color_name, cnts in cnts_list:
            for c in cnts:
                area = cv2.contourArea(c)
                if area > area_min:
                    area_min = area
                    max_cnt[0] = c
                    M = cv2.moments(c)
                    max_cnt[1] = int(M["m10"] / M["m00"])
                    max_cnt[2] = int(M["m01"] / M["m00"])
                    max_cnt[3] = color_name
        self.c = max_cnt
        # print(area_min, self.c[3]) ##

    def stop(self):
        self.stop_event.set()


Luck = {
    "Monday": {
        "Yellow": "Finance",
        "Green": "Work",
        "Red": "Misfortune",
        "Blue": "Love",
        "Black": "Work",
        "Cyan": "Love",
        "Purple": "Finance",
        "Brown": "Finance"},
    "Tuesday": {
        "Yellow": "Misfortune",
        "Green": "Work",
        "Red": "Love",
        "Blue": "Work",
        "Black": "Finance",
        "Cyan": "Work",
        "Purple": "Work",
        "Brown": "Finance"},
    "Wednesday": {
        "Yellow": "Work",
        "Green": "Love",
        "Red": "Misfortune",
        "Blue": "Work",
        "Black": "Love",
        "Cyan": "Finance",
        "Purple": "Misfortune",
        "Brown": "Work"},
    "Thursday": {
        "Yellow": "Work",
        "Green": "Love",
        "Red": "Finance",
        "Blue": "Work",
        "Black": "Work",
        "Cyan": "Work",
        "Purple": "Misfortune",
        "Brown": "Misfortune"},
    "Friday": {
        "Yellow": "Love",
        "Green": "Finance",
        "Red": "Finance",
        "Blue": "Work",
        "Black": "Misfortune",
        "Cyan": "Work",
        "Purple": "Misfortune",
        "Brown": "Work"},
    "Saturday": {
        "Yellow": "Work",
        "Green": "Misfortune",
        "Red": "Love",
        "Blue": "Finance",
        "Black": "Work",
        "Cyan": "Finance",
        "Purple": "Misfortune",
        "Brown": "Work"},
    "Sunday": {
        "Yellow": "Work",
        "Green": "Finance",
        "Red": "Work",
        "Blue": "Misfortune",
        "Black": "Love",
        "Cyan": "Misfortune",
        "Purple": "Finance",
        "Brown": "Finance"}}

pretrained_model = load_model("save_ckp_frozen.h5")
video_capture = cv2.VideoCapture(0)
ret, frames = video_capture.read()

thread = get_dress()
thread.start()


while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    # Show input image in the dimension of 512 x 512
    file = tf.image.resize_with_pad(
        frames, target_height=512, target_width=512)
    rgb = file.numpy()
    dummy = np.ones((rgb.shape[0], rgb.shape[1], 1))
    rgbx = np.concatenate((rgb, dummy * 255), axis=-1)
    output_img = rgbx / 255

    # Draw shirt area
    if thread.w != 0:
        cv2.rectangle(output_img, (thread.x, thread.y),
                      (thread.x + thread.w, thread.y + thread.h), (255, 0, 0), 3)
        if thread.c is not None and thread.c[1] is not None:
            # cv2.drawContours(output_img, [thread.c[0]], -1, (0, 255, 0), 3)
            cv2.circle(output_img, (thread.x +
                                    thread.c[1], thread.y +
                                    thread.c[2]), 7, (255, 255, 255), -
                       1)
            msg = Luck[datetime.now().strftime("%A")][thread.c[3]]
            cv2.putText(
                output_img,
                msg, (thread.x + thread.c[1] - 25,thread.y + thread.c[2] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            thread.c = None
    cv2.imshow('Shirt', output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        thread.stop()
        thread.join()
        break

video_capture.release()
cv2.destroyAllWindows()
