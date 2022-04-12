
#GoogleNet

import os
import time
from enum import IntFlag

import cv2
import numpy as np
#import vgamepad as vg
import win32gui
from mss import mss
from win32gui import FindWindow, GetWindowRect

from getkeys import key_check

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

starting_value = 1

while True:
	file_name = 'training_data-{}.npy'.format(starting_value)

	if os.path.isfile(file_name):
		print('File exists, moving along', starting_value)
		starting_value += 1
	else:
		print('File does not exist, starting fresh!', starting_value)

		break


def keys_to_output(keys):
	'''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
	output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

	if 'W' in keys and 'A' in keys:
		output = wa
	elif 'W' in keys and 'D' in keys:
		output = wd
	elif 'S' in keys and 'A' in keys:
		output = sa
	elif 'S' in keys and 'D' in keys:
		output = sd
	elif 'W' in keys:
		output = w
	elif 'S' in keys:
		output = s
	elif 'A' in keys:
		output = a
	elif 'D' in keys:
		output = d
	else:
		output = nk
	return output


def get_window_location(window_title = None):
	window_handle = FindWindow(None, window_title)
	window_rect = GetWindowRect(window_handle)
	return window_rect


def hough_lines(image):
	"""
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """

	return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=20, maxLineGap=20)

def draw_lines(image, lines, color=[128, 128, 128], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
	try:
		if make_copy:
			image = np.copy(image)  # don't want to modify the original
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(image, (x1, y1), (x2, y2), color, thickness)
		return image
	except:
		return image


# get video
def screen_record(image_coordinates):
	sct = mss()
	image = np.array(sct.grab(image_coordinates))
	return image

def get_vertices(image_rectangle):
	return np.array([[10, 600], [10, 300], [300, 200], [500, 200], [800, 300], [800, 600], ], np.int32)

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def image_processing(im_to_be_processed):
	# find yellow in an image
	hsv = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2GRAY)
	# Threshold of yellow in HSV space
	lower_yellow = np.array([20, 100, 100])
	upper_yellow = np.array([30, 255, 255])
	# preparing the mask to overlay
	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
	#yellow_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = masky)
	## find white in an image
	lower_white = np.array([0, 0, 168], dtype = np.uint8)
	upper_white = np.array([172, 111, 255], dtype = np.uint8)
	# preparing the mask to overlay
	maskw = cv2.inRange(hsv, lower_white, upper_white)
	#combine mask
	mask = cv2.bitwise_or(maskw, masky)
	#white_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = maskw)
	## look for gray or brown (dirt road surfaces)
	# preparing the mask to overlay
	yellow_and_white = cv2.cvtColor(cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = mask), cv2.COLOR_RGB2GRAY)
	yellow_and_white = cv2.GaussianBlur(yellow_and_white, (7, 7), 0)
	yellow_and_white = cv2.Canny(yellow_and_white, threshold1=200, threshold2=300)
	vertices = get_vertices([0, 80, 870, 600])
	#yellow_and_white = roi(yellow_and_white, [vertices])
	yellow_and_white = draw_lines(yellow_and_white, hough_lines(yellow_and_white))
	return  yellow_and_white



########################################################################################################################
#
# Start Runtime
#
########################################################################################################################
# initialize
def main(file_name, starting_value):
	file_name = file_name
	starting_value = starting_value
	training_data = []
	image_rectangle = get_window_location('3D Car Simulator - Play 3D Car Simulator on CrazyGames')
	for i in list(range(4))[::-1]:
		print(i + 1)
		time.sleep(1)

	last_time = time.time()
	paused = False
	print('STARTING!!!')
	print(image_rectangle)
	vertices = get_vertices([0, 80, 870, 600])
	"""
	gamepad = vg.VX360Gamepad()

	# press a button to wake the device up
	gamepad.press_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
	gamepad.update()
	time.sleep(0.5)
	gamepad.release_button(button = vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
	gamepad.update()
	time.sleep(0.5)

	image_rectangle = get_window_location('3D Car Simulator - Play 3D Car Simulator on CrazyGames')
	"""


	# Main Loop
	while True:
		im = screen_record(image_rectangle)
		#cv2.imshow('window2', im)
		# image processing
		screen = image_processing(im)
		# shows the image loaded into imshow
		#cv2.imshow('screen', im)

		screen = cv2.resize(screen, (480, 270))
		# run a color convert:
		screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2RGB)
		cv2.imshow('Processed Image', screen)
		keys = key_check()
		output = keys_to_output(keys)
		print(output)
		training_data.append([screen, output])

		if len(training_data) % 100 == 0:
			print(len(training_data))

			if len(training_data) == 500:
				np.save(file_name, training_data)
				print('SAVED')
				training_data = []
				starting_value += 1
				file_name = 'training_data-{}.npy'.format(starting_value)
		#keys = key_check()
		#cv2.imshow('original', draw_lines(cv2.bitwise_not(original_image,original_image) , hough_lines(processed_img)))
		#cv2.imshow('white screen', white_im)
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
		#gamepad.update()

main(file_name, starting_value)