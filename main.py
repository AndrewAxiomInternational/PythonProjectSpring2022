import ctypes.wintypes
import os
import time

import cv2
import numpy as np
import win32gui
from mss import mss


class XInputGamepad(ctypes.Structure):
	_fields_ = [
		('wButtons', ctypes.wintypes.WORD),
		('bLeftTrigger', ctypes.wintypes.BYTE),
		('bRightTrigger', ctypes.wintypes.BYTE),
		('sThumbLX', ctypes.wintypes.SHORT),
		('sThumbLY', ctypes.wintypes.SHORT),
		('sThumbRX', ctypes.wintypes.SHORT),
		('sThumbRY', ctypes.wintypes.SHORT)
	]


class XInputState(ctypes.Structure):
	_fields_ = [('dwPacketNumber', ctypes.wintypes.DWORD),('Gamepad', XInputGamepad),]


class XInputVibration(ctypes.Structure):
	_fields_ = [('wLeftMotorSpeed', ctypes.wintypes.WORD),('wRightMotorSpeed', ctypes.wintypes.WORD)]


def get_window_location(window_title = None):
	if window_title:
		hwnd = win32gui.FindWindow(None, window_title)
		if hwnd:
			win32gui.SetForegroundWindow(hwnd)
			x_int, y_int, x1_int, y1_int = win32gui.GetClientRect(hwnd)
			x_int, y_int = win32gui.ClientToScreen(hwnd, (x_int, y_int))
			x1_int, y1_int = win32gui.ClientToScreen(hwnd, (x1_int - x_int, y1_int - y_int))
		else:
			print('Window not found!')
		return x_int, y_int, x1_int, y1_int


# get video
def screen_record(xsr, ysr, x1sr, y1sr):
	sct = mss()
	image = np.array(sct.grab((xsr, ysr, x1sr, y1sr)))
	return image


def image_processing(im_to_be_processed):
	# find yellow in an image
	hsv = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2GRAY)
	# Threshold of yellow in HSV space
	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])
	# preparing the mask to overlay
	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = masky)
	## find white in an image
	lower_white = np.array([0, 0, 168], dtype = np.uint8)
	upper_white = np.array([172, 111, 255], dtype = np.uint8)
	# preparing the mask to overlay
	maskw = cv2.inRange(hsv, lower_white, upper_white)
	white_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = maskw)
	## look for gray or brown (dirt road surfaces)
	# preparing the mask to overlay
	maskgr = cv2.inRange(gray, 100, 255)
	im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = maskgr)
	return im, yellow_im, white_im


########################################################################################################################
#
# Start Runtime
#
########################################################################################################################
# initialize
if __name__ == '__main__':
	api = ctypes.windll.xinput1_4
	state = XInputState()
	gamepad_number = 0

	while True:
		api.XInputGetState(
				ctypes.wintypes.WORD(gamepad_number),
				ctypes.pointer(state)
		)
		print(state.dwPacketNumber)
		time.sleep(0.5)
		os.system('cls')
x, y, x1, y1 = get_window_location('Grand Theft Auto V')

# Main Loop
while True:
	im = screen_record(x, y, x1, y1)
	# image processing
	im, yellow_im, white_im = image_processing(im)
	# shows the image loaded into imshow
	cv2.imshow('screen', white_im)
	# this will break the loop when 'q' is pressed
	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
