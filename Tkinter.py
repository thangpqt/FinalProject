import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from time import sleep
from threading import Thread

window = Tk()
window.title("Phạm Quốc Thắng 19146393")
video = cv2.VideoCapture(0)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

canvas = Canvas(window, width = canvas_w, height= canvas_h , bg= "blue")
canvas.pack()

bw = 0

def handleBW():
    global bw
    bw = 1 - bw

button = Button(window,text = "Start detect...", command=handleBW)
button.pack()

photo = None
count = 0

def send_to_server():
    global button
    sleep(10)
    button.configure(text="loading")
    return

def update_frame():
    global canvas, photo, bw, count
    # Doc tu camera
    ret, frame = video.read()
    # Ressize
    #frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    # Chuyen he mau
    if bw==0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        label = "SWING BODY"
        n_time_steps = 10
        lm_list = []
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        mpDraw = mp.solutions.drawing_utils
        model = tf.keras.models.load_model("model.h5")
        def make_landmark_timestep(results):
            c_lm = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
                c_lm.append(lm.visibility)
            return c_lm
        def draw_landmark_on_image(mpDraw, results, img):
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, ( 0,255, 0), cv2.FILLED)
            return img
        def draw_class_on_image(label, img):
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (0,0,255)
            thickness = 2
            lineType = 2
            cv2.putText(img, label,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            return img
        def detect(model, lm_list):
            global label
            lm_list = np.array(lm_list)
            lm_list = np.expand_dims(lm_list, axis=0)
            print(lm_list.shape)
            lm_list = lm_list.reshape(1, 1320)
            lm_list = lm_list.astype('float32')
            results = model.predict(lm_list)
            print(results)
            if np.argmax(results) == 1:
                label = "SWING BODY"
            elif np.argmax(results) == 0:
                label = "SWING HAND"
            return label
        i = 0
        warmup_frames = 60
        while True:
            success, img = video.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            i = i + 1
            if i > warmup_frames:
                print("Start detect....")

                if results.pose_landmarks:
                    c_lm = make_landmark_timestep(results)

                    lm_list.append(c_lm)
                    if len(lm_list) == n_time_steps:
                        # predict
                        t1 = threading.Thread(target=detect, args=(model, lm_list,))
                        t1.start()
                        lm_list = []
                    img = draw_landmark_on_image(mpDraw, results, img)
            img = draw_class_on_image(label, img)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    # Convert hanh image TK
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    # Show
    canvas.create_image(0,0, image = photo, anchor=tkinter.NW)

    count = count +1
    if count%10==0:
        #send_to_server()
        thread = Thread(target=send_to_server)
        thread.start()

    window.after(60, update_frame)

update_frame()

window.mainloop()
