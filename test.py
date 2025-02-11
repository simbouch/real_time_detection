from utils.opencv.capturCam import CapturCam


x = CapturCam()
frame=x.get_frame()
frame = x.color(frame)
x.picture(frame)

print("Done")