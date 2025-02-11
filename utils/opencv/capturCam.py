import cv2
import numpy as np


class CapturCam:
    def __init__(self, camera=0):
        self.camera = cv2.VideoCapture(camera)

    def get_frame(self):
        success, frame = self.camera.read()
        if not success:
            raise Exception("Could not read from camera")
        return frame

    def release(self):
        self.camera.release()
    
    def color(self, frame):
        return cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    
    def picture(self, frame):
        cv2.imshow('frame', frame)
        