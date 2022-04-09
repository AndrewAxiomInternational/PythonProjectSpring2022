import cv2

import ImageImport
import ImageProcessing

x, y, x1, y1 = ImageImport.get_window_location('Grand Theft Auto V')
#get image
while True:
	im = ImageImport.screen_record(x, y, x1, y1)
	# image processing
	im, yellow_im, white_im=ImageProcessing.image_processing(im)
	cv2.imshow('screen', im)
	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
