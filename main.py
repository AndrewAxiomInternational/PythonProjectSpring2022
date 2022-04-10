import time
from enum import IntFlag

import cv2
import numpy as np
import vgamepad as vg
# import win32gui
from mss import mss
from win32gui import FindWindow, GetWindowRect


class XUSB_BUTTON(IntFlag):
	"""
    Possible XUSB report buttons.
    """
	XUSB_GAMEPAD_DPAD_UP = 0x0001  # phone up
	XUSB_GAMEPAD_DPAD_DOWN = 0x0002  # phone down
	XUSB_GAMEPAD_DPAD_LEFT = 0x0004  # Phone left
	XUSB_GAMEPAD_DPAD_RIGHT = 0x0008  # phone right
	XUSB_GAMEPAD_START = 0x0010  # pause menu
	XUSB_GAMEPAD_BACK = 0x0020  # interaction menu
	XUSB_GAMEPAD_LEFT_THUMB = 0x0040  # horn
	XUSB_GAMEPAD_RIGHT_THUMB = 0x0080  # look behind
	XUSB_GAMEPAD_LEFT_SHOULDER = 0x0100  # aim weapon
	XUSB_GAMEPAD_RIGHT_SHOULDER = 0x0200  # handbrake
	XUSB_GAMEPAD_GUIDE = 0x0400  #
	XUSB_GAMEPAD_A = 0x1000  # duck
	XUSB_GAMEPAD_B = 0x2000  #
	XUSB_GAMEPAD_X = 0x4000  #
	XUSB_GAMEPAD_Y = 0x8000  # exit vehicle


def get_window_location(window_title = None):
	window_handle = FindWindow(None, window_title)
	window_rect = GetWindowRect(window_handle)
	return window_rect


# get video
def screen_record(image_coordinates):
	sct = mss()
	image = np.array(sct.grab(image_coordinates))
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

	image_rectangle = get_window_location('Grand Theft Auto V')
	# Main Loop
	while True:
		im = screen_record(image_rectangle)
		# image processing
		im, yellow_im, white_im = image_processing(im)
		# shows the image loaded into imshow
		cv2.imshow('screen', im)
		# this will break the loop when 'q' is pressed
		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			cv2.destroyAllWindows()
			break

		# POSSIBLE CONTROLLER INPUTS
		# gamepad.press_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A) # button list in XUSB_BUTTON
		# gamepad.release_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A) # button list in XUSB_BUTTON
		# gamepad.left_trigger_float(value_float = 0.5)# 0 to 1
		# gamepad.right_trigger_float(value_float = 0.5) # 0 to 1
		# gamepad.left_joystick_float(x_value_float = 0.0, y_value_float = 0.2) # -1 to 1
		# gamepad.right_joystick_float(x_value_float = -1.0, y_value_float = 1.0)# -1 to 1
		# controller update
		gamepad.update()
