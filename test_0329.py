import tkinter as tk
import cv2
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(0)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Start the video stream
        while True:
            # Get a frame from the video source
            ret, frame = self.vid.read()

            if ret:
                # Convert the color space from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.delete('img')
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags='img')

            # Update the window after delay milliseconds
            self.window.update()
            self.window.after(15)

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV")