import numpy as np
import cv2
import keyboard

PUPIL_THRESHOLD = 125
WINK_THRESHOLD = 45
MOVING_AVG_LENGTH = 5
WINK_PROB_THRESHOLD = 0.2
SHOW_VIDEO = False

def press_key(key):
	try:
		keyboard.send(key)
	except ImportError as e:
		print('Failed to send key press:', e)

def wink_started(eye):
	print('wink stated', eye)
	if eye == 'left':
		press_key('left')
	elif eye == 'right':
		press_key('right')

def wink_stopped(eye):
	print('wink stopped', eye)
	if eye == 'left':
		pass
	elif eye == 'right':
		pass

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

left_wink = []
right_wink = []
left_winking = False
right_winking = False

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
	gray = ((gray - gray.min()) * (255 / (gray.max() - gray.min()))).astype(np.uint8)

	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	if len(faces) == 0:
		continue
	fx, fy, fw, fh = faces[0]
	face = gray[fy:fy+fw,fx:fx+fh]

	eyes = eye_cascade.detectMultiScale(face, 1.1, 4)
	left_eyes = []
	right_eyes = []
	for (ex, ey, ew, eh) in eyes:
		if ey < fh / 2:
			if ex > fw / 2:
				left_eyes.append((ex, ey, ew, eh))
			else:
				right_eyes.append((ex, ey, ew, eh))

	area = lambda dims: dims[2] * dims[3]

	left_eye = None
	if len(left_eyes) > 0:
		left_eye = left_eyes[np.argmax(list(map(area, left_eyes)))]
	right_eye = None
	if len(right_eyes) > 0:
		right_eye = right_eyes[np.argmax(list(map(area, right_eyes)))]

	for i, curr in enumerate([left_eye, right_eye]):
		if curr is None:
			continue

		ex, ey, ew, eh = curr
		eye = face[ey:ey+ew,ex:ex+eh]
		eye_mask = np.zeros(eye.shape, dtype=np.bool)
		for y, x in np.argwhere(eye < PUPIL_THRESHOLD):
			if x >= ew / 4 and x < ew / 4 * 3 and y >= eh / 4 and y < eh / 4 * 3:
				eye_mask[y,x] = True

		if i == 0:
			left_wink.append(False)
		else:
			right_wink.append(False)

		x_ax = np.argwhere(eye_mask.sum(axis=0) > 0)
		y_ax = np.argwhere(eye_mask.sum(axis=1) > 0)
		if len(x_ax) > 0 and len(y_ax) > 0:
			if (x_ax.max() - x_ax.min()) + (y_ax.max() - y_ax.min()) > WINK_THRESHOLD:
				if SHOW_VIDEO:
					cv2.rectangle(img, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 0, 255), 2)
				if i == 0:
					left_wink[-1] = True
				elif i == 1:
					right_wink[-1] = True

	while len(left_wink) > MOVING_AVG_LENGTH:
		left_wink.pop(0)
	while len(right_wink) > MOVING_AVG_LENGTH:
		right_wink.pop(0)

	if np.mean(left_wink) > WINK_PROB_THRESHOLD:
		if left_winking == False:
			wink_started('left')
		left_winking = True
	elif left_winking:
		left_winking = False
		wink_stopped('left')

	if np.mean(right_wink) > WINK_PROB_THRESHOLD:
		if right_winking == False:
			wink_started('right')
		right_winking = True
	elif right_winking:
		right_winking = False
		wink_stopped('right')

	if SHOW_VIDEO:
		cv2.imshow('img', img)
		if cv2.waitKey(1) == 27:
			break

cap.release()
cv2.destroyAllWindows()
