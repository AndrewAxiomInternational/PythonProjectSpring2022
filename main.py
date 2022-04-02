import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import win32gui
from mss import mss

## get video
def screen_record(window_title=None):
	if window_title:
		hwnd = win32gui.FindWindow(None, window_title)
		if hwnd:
			win32gui.SetForegroundWindow(hwnd)
			x, y, x1, y1 = win32gui.GetClientRect(hwnd)
			x, y = win32gui.ClientToScreen(hwnd, (x, y))
			x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
			#im = pyautogui.screenshot(region = (x, y, x1, y1))
			sct=mss()
			im=sct.grab((x, y, x1, y1))
			return im
		else:
			print('Window not found!')
	else:
		im = pyautogui.screenshot()
		return im


while True:

	im = screen_record('Grand Theft Auto V')
	cv2.imshow('screen', np.array(im))

	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
