import cv2
import numpy as np
import pygame
import win32gui
from mss import mss


def get_window_location(window_title = None):
	if window_title:
		hwnd = win32gui.FindWindow(None, window_title)
		if hwnd:
			win32gui.SetForegroundWindow(hwnd)
			x, y, x1, y1 = win32gui.GetClientRect(hwnd)
			x, y = win32gui.ClientToScreen(hwnd, (x, y))
			x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
		# im = pyautogui.screenshot(region = (x, y, x1, y1))
		else:
			print('Window not found!')
		return x, y, x1, y1


# get video
def screen_record(xsr, ysr, x1sr, y1sr):
	sct = mss()
	image = np.array(sct.grab((xsr, ysr, x1sr, y1sr)))
	return image


def image_processing(imToBeProcessed):
	# find yellow in an image
	hsv = cv2.cvtColor(imToBeProcessed, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(imToBeProcessed, cv2.COLOR_BGR2GRAY)
	# Threshold of yellow in HSV space
	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])
	# preparing the mask to overlay
	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_im = cv2.bitwise_and(imToBeProcessed, imToBeProcessed, mask = masky)
	## find white in an image
	lower_white = np.array([0, 0, 168], dtype = np.uint8)
	upper_white = np.array([172, 111, 255], dtype = np.uint8)
	# preparing the mask to overlay
	maskw = cv2.inRange(hsv, lower_white, upper_white)
	white_im = cv2.bitwise_and(imToBeProcessed, imToBeProcessed, mask = maskw)
	## look for gray or brown (dirt road surfaces)
	# preparing the mask to overlay
	maskgr = cv2.inRange(gray, 100, 255)
	im = cv2.bitwise_and(imToBeProcessed, imToBeProcessed, mask = maskgr)
	return im, yellow_im, white_im
########################################################################################################################
#
# Start Runtime
#
########################################################################################################################
#initialize
pygame.init()
x, y, x1, y1 = get_window_location('Grand Theft Auto V')

# Main Loop
while True:
	im = screen_record(x, y, x1, y1)
	# image processing
	im, yellow_im, white_im=image_processing(im)
	#shows the image loaded into imshow
	cv2.imshow('screen', white_im)
	# this will break the loop when 'q' is pressed
	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
