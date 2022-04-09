import time
from enum import IntFlag

import cv2
import numpy as np
import vgamepad as vg
import win32gui
from mss import mss


class XUSB_BUTTON(IntFlag):
    """
    Possible XUSB report buttons.
    """
    XUSB_GAMEPAD_DPAD_UP = 0x0001
    XUSB_GAMEPAD_DPAD_DOWN = 0x0002
    XUSB_GAMEPAD_DPAD_LEFT = 0x0004
    XUSB_GAMEPAD_DPAD_RIGHT = 0x0008
    XUSB_GAMEPAD_START = 0x0010
    XUSB_GAMEPAD_BACK = 0x0020
    XUSB_GAMEPAD_LEFT_THUMB = 0x0040
    XUSB_GAMEPAD_RIGHT_THUMB = 0x0080
    XUSB_GAMEPAD_LEFT_SHOULDER = 0x0100
    XUSB_GAMEPAD_RIGHT_SHOULDER = 0x0200
    XUSB_GAMEPAD_GUIDE = 0x0400
    XUSB_GAMEPAD_A = 0x1000
    XUSB_GAMEPAD_B = 0x2000
    XUSB_GAMEPAD_X = 0x4000
    XUSB_GAMEPAD_Y = 0x8000

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
	gamepad = vg.VX360Gamepad()

	# press a button to wake the device up
	gamepad.press_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
	gamepad.update()
	time.sleep(0.5)
	gamepad.release_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
	gamepad.update()
	time.sleep(0.5)

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

		# POSSIBLE CONTROLLER INPUTS
		#gamepad.press_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
		#gamepad.release_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
		#gamepad.left_trigger_float(value_float = 0.5)
		#gamepad.right_trigger_float(value_float = 0.5)
		#gamepad.left_joystick_float(x_value_float = 0.0, y_value_float = 0.2)
		#gamepad.right_joystick_float(x_value_float = -1.0, y_value_float = 1.0)
		# controller update
		gamepad.update()
