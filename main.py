import cv2
import numpy as np
# from PIL import ImageGrab
import win32gui
from mss import mss


## get video

def get_window_location(window_title=None):
	if window_title:
		hwnd = win32gui.FindWindow(None, window_title)
		if hwnd:
			win32gui.SetForegroundWindow(hwnd)
			x, y, x1, y1 = win32gui.GetClientRect(hwnd)
			x, y = win32gui.ClientToScreen(hwnd, (x, y))
			x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
			#im = pyautogui.screenshot(region = (x, y, x1, y1))
		else:
			print('Window not found!')
		return x, y, x1, y1

def screen_record(x, y, x1, y1):
	sct = mss()
	im = np.array(sct.grab((x, y, x1, y1)))
	return im
def image_processing(im):
## find yellow in an image
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# Threshold of yellow in HSV space
	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])
	# preparing the mask to overlay
	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_im = cv2.bitwise_and(im, im, mask = masky)
## find white in an image
	lower_white = np.array([0, 0, 168], dtype=np.uint8)
	upper_white = np.array([172, 111, 255], dtype=np.uint8)
	# preparing the mask to overlay
	maskw = cv2.inRange(hsv, lower_white, upper_white)
	white_im = cv2.bitwise_and(im, im, mask = maskw)
## look for gray or brown (dirt road surfaces)
	# preparing the mask to overlay
	maskgr = cv2.inRange(gray, 100, 255)
	im = cv2.bitwise_and(im, im, mask = maskgr)

	return im, yellow_im, white_im


# get image box while allowing gta5 not to be in focus
x, y, x1, y1 = get_window_location('Grand Theft Auto V')
#get image
while True:
	im = screen_record(x, y, x1, y1)
	# image processing
	im, yellow_im, white_im=image_processing(im)
	#cv2.imshow('screen', np.array(im))

	cv2.imshow('screen', im)


	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
