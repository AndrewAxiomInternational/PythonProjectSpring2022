import numpy as np
from mss import mss

from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from Archive.models import inception_v3 as googlenet
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode, mean
import numpy as np
from motion import motion_detection
from win32gui import FindWindow, GetWindowRect

GAME_WIDTH = 800
GAME_HEIGHT = 600

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

t_time = 0.25


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def image_processing2(im_to_be_processed):
    # find yellow in an image
    hsv = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im_to_be_processed, cv2.COLOR_BGR2GRAY)
    # Threshold of yellow in HSV space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # preparing the mask to overlay
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # yellow_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = masky)
    ## find white in an image
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)
    # preparing the mask to overlay
    maskw = cv2.inRange(hsv, lower_white, upper_white)
    # combine mask
    mask = cv2.bitwise_or(maskw, masky)
    # white_im = cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask = maskw)
    ## look for gray or brown (dirt road surfaces)
    # preparing the mask to overlay
    yellow_and_white = cv2.cvtColor(cv2.bitwise_and(im_to_be_processed, im_to_be_processed, mask=mask),
                                    cv2.COLOR_RGB2GRAY)
    yellow_and_white = cv2.GaussianBlur(yellow_and_white, (7, 7), 0)
    yellow_and_white = cv2.Canny(yellow_and_white, threshold1=200, threshold2=300)
    # vertices = get_vertices([0, 80, 870, 600])
    # yellow_and_white = roi(yellow_and_white, [vertices])
    yellow_and_white = draw_lines(yellow_and_white, hough_lines(yellow_and_white))
    return yellow_and_white


def get_vertices(image_rectangle):
    return np.array([[10, 600], [10, 300], [300, 200], [500, 200], [800, 300], [800, 600], ], np.int32)


def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=20, maxLineGap=20)


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


def get_window_location(window_title=None):
    window_handle = FindWindow(None, window_title)
    window_rect = GetWindowRect(window_handle)
    return window_rect


def screen_record(image_coordinates):
    sct = mss()
    image = np.array(sct.grab(image_coordinates))
    return image


model = googlenet(WIDTH, HEIGHT, 3, LR, output=9)
MODEL_NAME = 'train_model'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    image_rectangle = get_window_location('Grand Theft Auto V')
    im = screen_record(image_rectangle)
    screen = cv2.resize(im, (480, 270))
    prev = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


    t_minus = prev
    t_now = prev
    t_plus = prev

    while (True):

        if not paused:
            im = screen_record(image_rectangle)
            screen = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            last_time = time.time()

            delta_count_last = motion_detection(t_minus, t_now, t_plus)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen
            t_plus = cv2.blur(t_plus, (4, 4))

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0]
            print("predicting key", prediction)
            prediction = np.array(prediction) * np.array([0.5, 2, 4, 1, 1.8, 1.8, 0.5, 0.5, 0.2])
            mode_choice = np.argmax(prediction)
            #print("predicting mode choice", mode_choice)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'

            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'

            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            motion_log.append(delta_count_last)
            motion_avg = round(mean(motion_log), 3)
            print('loop took {} seconds. Motion: {}. Choice: {}'.format(round(time.time() - last_time, 3), motion_avg,
                                                                        choice_picked))

            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0, 4)

                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 2:
                    reverse_left()
                    time.sleep(random.uniform(1, 2))
                    forward_right()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 3:
                    reverse_right()
                    time.sleep(random.uniform(1, 2))
                    forward_left()
                    time.sleep(random.uniform(1, 2))

                for i in range(log_len - 2):
                    del motion_log[0]

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()
