import cv2
import numpy as np
import dlib
import math
from math import sqrt, pow, acos
 
def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return acos(cos)

# initialize the face detector, facial landmarks detector, and the video stream
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
video_stream = cv2.VideoCapture(0)

while True:
    # capture the frame from the video stream
    ret, frame = video_stream.read()

    # convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the faces in the grayscale frame
    faces = face_detector(gray)

    # loop through each face and plot circles on the eyes and face
    for face in faces:
        # get the facial landmarks for the face
        landmarks = landmark_detector(gray, face)
        """
        {
            IdxRange jaw;       // [0 , 16]
            IdxRange rightBrow; // [17, 21]
            IdxRange leftBrow;  // [22, 26]
            IdxRange nose;      // [27, 35]
            IdxRange rightEye;  // [36, 41]
            IdxRange leftEye;   // [42, 47]
            IdxRange mouth;     // [48, 59]
            IdxRange mouth2;    // [60, 67]
        }
        """
        # plot circles on the eyes
        
        avg_left_eye_x = int((landmarks.part(36).x + landmarks.part(39).x) / 2)
        avg_left_eye_y = int((landmarks.part(36).y + landmarks.part(39).y) / 2)
        avg_right_eye_x = int((landmarks.part(45).x + landmarks.part(42).x) / 2)
        avg_right_eye_y = int((landmarks.part(45).y + landmarks.part(42).y) / 2)
        '''
        left_eye_center = np.array([avg_left_eye_x, avg_left_eye_y])
        right_eye_center = np.array([avg_right_eye_x, avg_right_eye_y])
        left_eye_radius = np.sqrt((landmarks.part(37).x - landmarks.part(41).x) ** 2 + (landmarks.part(37).y - landmarks.part(41).y) ** 2)
        right_eye_radius = np.sqrt((landmarks.part(43).x - landmarks.part(47).x) ** 2 + (landmarks.part(43).y - landmarks.part(47).y) ** 2)
        cv2.circle(frame, tuple(left_eye_center), int(left_eye_radius), (0, 255, 0), 2)
        cv2.circle(frame, tuple(right_eye_center), int(right_eye_radius), (0, 255, 0), 2)
        
        w = abs(landmarks.part(36).x - landmarks.part(45).x) + 20 
        h = abs(landmarks.part(46).y - landmarks.part(37).y) * 5
        x = landmarks.part(36).x - 10
        y = landmarks.part(37).y - 10
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        '''
        a = [1, 0]
        b = [(avg_right_eye_x - avg_left_eye_x), (avg_right_eye_y - avg_left_eye_y)]
        
        two_eyes_angle = angle_of_vector(a, b)
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

        '''
        x1 = (landmarks.part(36).x - 20)
        y1 = (landmarks.part(37).y - 20)
        x2 = (landmarks.part(36).x - 20)
        y2 = (landmarks.part(41).y + 20)
        x3 = (landmarks.part(45).x + 20)
        y3 = (landmarks.part(44).y - 20)
        x4 = (landmarks.part(45).x + 20)
        y4 = (landmarks.part(46).y + 20)
        '''        
        cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必須是array數組
        rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (寬,高), 旋轉角度）
        box = cv2.boxPoints(rect) # 獲取最小外接矩形的4個頂點座標(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        
        # plot circle around the face
        #face_center = np.array([(face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2])
        face_center = np.array([landmarks.part(29).x, landmarks.part(29).y])
        #face_radius = int(np.sqrt((face.right() - face.left()) ** 2 + (face.bottom() - face.top()) ** 2) / 2)
        face_radius = int(np.sqrt(((landmarks.part(29).x - landmarks.part(8).x) ** 2) + ((landmarks.part(29).y - landmarks.part(8).y) ** 2)))
        cv2.circle(frame, tuple(face_center), face_radius, (0, 0, 255), 2)

    # show the frame with circles plotted on the eyes and face
    cv2.imshow("Frame", frame)

    # check for key press
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

# release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()
