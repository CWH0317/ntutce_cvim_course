import tkinter as tk
import cv2
from PIL import Image, ImageTk
import tkinter.font as tkFont
import numpy as np
import dlib
import math
from math import sqrt, pow, acos

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(0)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(side=tk.LEFT)
        
        #setting window size
        width=1000
        height=600
        screenwidth = window.winfo_screenwidth()
        screenheight = window.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        window.geometry(alignstr)
        window.resizable(width=False, height=False)

        self.draw_eyes = False
        self.draw_face = False

        self.GCheckBox_371=tk.Checkbutton(window)
        ft = tkFont.Font(family='Times',size=16)
        self.GCheckBox_371["font"] = ft
        self.GCheckBox_371["fg"] = "#333333"
        self.GCheckBox_371["justify"] = "center"
        self.GCheckBox_371["text"] = "Eyes"
        self.GCheckBox_371.place(x=800,y=200,width=165,height=64)
        self.GCheckBox_371["offvalue"] = "0"
        self.GCheckBox_371["onvalue"] = "1"
        self.GCheckBox_371["command"] = self.GCheckBox_371_command

        self.GCheckBox_175=tk.Checkbutton(window)
        ft = tkFont.Font(family='Times',size=16)
        self.GCheckBox_175["font"] = ft
        self.GCheckBox_175["fg"] = "#333333"
        self.GCheckBox_175["justify"] = "center"
        self.GCheckBox_175["text"] = "Face"
        self.GCheckBox_175.place(x=850,y=340,width=70,height=25)
        self.GCheckBox_175["offvalue"] = "0"
        self.GCheckBox_175["onvalue"] = "1"
        self.GCheckBox_175["command"] = self.GCheckBox_175_command
        

        face_detector = dlib.get_frontal_face_detector()
        landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Start the video stream
        while True:
            # Get a frame from the video source
            ret, frame = self.vid.read()
            # convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect the faces in the grayscale frame
            faces = face_detector(gray)
            # loop through each face and plot circles on the eyes and face
            for face in faces:
                # get the facial landmarks for the face
                landmarks = landmark_detector(gray, face)

                avg_left_eye_x = int((landmarks.part(36).x + landmarks.part(39).x) / 2)
                avg_left_eye_y = int((landmarks.part(36).y + landmarks.part(39).y) / 2)
                avg_right_eye_x = int((landmarks.part(45).x + landmarks.part(42).x) / 2)
                avg_right_eye_y = int((landmarks.part(45).y + landmarks.part(42).y) / 2)
                
                a = [1, 0]
                b = [(avg_right_eye_x - avg_left_eye_x), (avg_right_eye_y - avg_left_eye_y)]
                
                two_eyes_angle = self.angle_of_vector(a, b)
                #print(two_eyes_angle)
                if b[1] >= 0 :
                    angle_1 = two_eyes_angle + (135 /180 * math.pi) # 135
                    angle_2 = two_eyes_angle + (-135 /180 * math.pi) # -135
                    angle_3 = two_eyes_angle + (45 /180 * math.pi) # 45
                    angle_4 = two_eyes_angle + (-45 /180 * math.pi) # -45
                else:
                    angle_1 = -two_eyes_angle + (135 /180 * math.pi) # 135
                    angle_2 = -two_eyes_angle + (-135 /180 * math.pi) # -135
                    angle_3 = -two_eyes_angle + (45 /180 * math.pi) # 45
                    angle_4 = -two_eyes_angle + (-45 /180 * math.pi) # -45
                #print(angle_1,angle_2,angle_3,angle_4)
                x1 = int(avg_left_eye_x + 40 * math.cos(angle_1))
                y1 = int(avg_left_eye_y + 40 * math.sin(angle_1))
                x2 = int(avg_left_eye_x + 40 * math.cos(angle_2))
                y2 = int(avg_left_eye_y + 40 * math.sin(angle_2))
                x3 = int(avg_right_eye_x + 40 * math.cos(angle_3))
                y3 = int(avg_right_eye_y + 40 * math.sin(angle_3))
                x4 = int(avg_right_eye_x + 40 * math.cos(angle_4))
                y4 = int(avg_right_eye_y + 40 * math.sin(angle_4))
            
                cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必須是array數組
                rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (寬,高), 旋轉角度）
                box = cv2.boxPoints(rect) # 獲取最小外接矩形的4個頂點座標(ps: cv2.boxPoints(rect) for OpenCV 3.x)
                box = np.int0(box)
                if self.draw_eyes:  
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                
                # plot circle around the face
                #face_center = np.array([(face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2])
                face_center = np.array([landmarks.part(29).x, landmarks.part(29).y])
                #face_radius = int(np.sqrt((face.right() - face.left()) ** 2 + (face.bottom() - face.top()) ** 2) / 2)
                face_radius = int(np.sqrt(((landmarks.part(29).x - landmarks.part(8).x) ** 2) + ((landmarks.part(29).y - landmarks.part(8).y) ** 2)))
                if self.draw_face:
                    cv2.circle(frame, tuple(face_center), face_radius, (0, 0, 255), 2)
                    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting image
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

            #refreshing the window
            self.window.update()
        
        # If we've gotten to the end of the video, release the capture
        self.vid.release()

        # Close the window
        self.window.destroy()

            
    def GCheckBox_371_command(self):
        self.draw_eyes = not self.draw_eyes

    def GCheckBox_175_command(self):
        self.draw_face = not self.draw_face
    
    def angle_of_vector(self, v1, v2):
        pi = 3.1415
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        return acos(cos)
            
# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV")