import cv2
import numpy as np

class ImageProcessing:
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
		lower_white = np.array([0, 0, 168], dtype = np.uint8)
		upper_white = np.array([172, 111, 255], dtype = np.uint8)
		# preparing the mask to overlay
		maskw = cv2.inRange(hsv, lower_white, upper_white)
		white_im = cv2.bitwise_and(im, im, mask = maskw)
		## look for gray or brown (dirt road surfaces)
		# preparing the mask to overlay
		maskgr = cv2.inRange(gray, 100, 255)
		im = cv2.bitwise_and(im, im, mask = maskgr)

		return im, yellow_im, white_im