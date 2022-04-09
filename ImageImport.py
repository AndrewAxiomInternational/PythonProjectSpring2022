import numpy as np
import win32gui
from mss import mss


class ImageImport:
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

	## get video
	def screen_record(x, y, x1, y1):
		sct = mss()
		im = np.array(sct.grab((x, y, x1, y1)))
		return im