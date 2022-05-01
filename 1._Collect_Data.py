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

def screen_record(image_coordinates):
	sct = mss()
	image = np.array(sct.grab(image_coordinates))
	return image

def get_vertices(image_rectangle):
	return np.array([[10, 600], [10, 300], [300, 200], [500, 200], [800, 300], [800, 600], ], np.int32)


def image_processing(im_to_be_processed):
	gray = cv2.resize(im_to_be_processed, (480, 270))
	return  gray



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
	image_rectangle = get_window_location('Grand Theft Auto V')

	for i in list(range(4))[::-1]:wwwwwwwww
		print(i + 1)
		time.sleep(1)

	last_time = time.time()
	paused = False
	print('STARTING!!!')
	print(image_rectangle)
	vertices = get_vertices([0, 80, 870, 600])
	while True:
		im = screen_record(image_rectangle)
		screen = cv2.resize(im, (480, 270))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
		#cv2.imshow('Neural Net screen', screen)
		keys = key_check()
		output = keys_to_output(keys)
		training_data.append([screen, output])

		if len(training_data) % 100 == 0:
			print(len(training_data))

			if len(training_data) == 500:
				np.save(file_name, training_data)
				print('SAVED')
				training_data = []
				starting_value += 1
				file_name = 'training_data-{}.npy'.format(starting_value)

		# this will break the loop when 'q' is pressed
		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			cv2.destroyAllWindows()
			break
		keys = key_check()
		if 'T' in keys:
			if paused:
				paused = False
				print('unpaused!')
				time.sleep(1)
			else:
				print('Pausing!')
				paused = True
				time.sleep(1)


main(file_name, starting_value)