import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import win32gui

## get video
def screen_record(window_title=None):
	if window_title:
		hwnd = win32gui.FindWindow(None, window_title)
		if hwnd:
			win32gui.SetForegroundWindow(hwnd)
			x, y, x1, y1 = win32gui.GetClientRect(hwnd)
			x, y = win32gui.ClientToScreen(hwnd, (x, y))
			x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
			im = pyautogui.screenshot(region = (x, y, x1, y1))
			return im
		else:
			print('Window not found!')
	else:
		im = pyautogui.screenshot()
		return im




im = screen_record('Grand Theft Auto V')
if im:
    im.show()