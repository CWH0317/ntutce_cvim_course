import cv2
import tkinter as tk

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(width=self.width, height=self.height)
            img.put((frame // 256).flatten(), to=(self.width - 1, self.height - 1, 3))
            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)

        self.window.after(self.delay, self.update)

App(tk.Tk(), "Tkinter and OpenCV")