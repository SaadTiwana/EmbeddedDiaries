
# This code is Derived from code in https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/


# USAGE:
# Activate environment:
# 	activate OpenCV-master-py3
#
# Run Code:
# python multi_object_tracking.py --source file --video videos/nascar.mp4 --tracker csrt
# python multi_object_tracking.py --source camera --tracker csrt


# Source types: network, file, camera

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
print(cv2.__version__)

import requests
import numpy as np

from pynput import keyboard
from pynput.keyboard import Key, Listener

from imutils.video import FPS


# url for Android camera on network
url = "http://192.168.0.167:8080/shot.jpg"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=str, help="Type of Video Source")
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="mosse", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,				# good. Works with rotation change
	"kcf": cv2.TrackerKCF_create,				# crap
	"boosting": cv2.TrackerBoosting_create,		# crap
	"mil": cv2.TrackerMIL_create,				# crap
	"tld": cv2.TrackerTLD_create,				# no bad. Works up to specific rotation
	"medianflow": cv2.TrackerMedianFlow_create,	# ok
	"mosse": cv2.TrackerMOSSE_create			# crap
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

	
if (args["source"]=="network"):
	# do nothing
	print("[INFO] network stream")

elif (args["source"]=="file"):
	vs = cv2.VideoCapture(args["video"])
	
elif (args["source"]=="camera"):
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=1).start()
	vs = cv2.VideoCapture(1)
	vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
	#vs.set(15, 1000)
	time.sleep(1.0)
	
else:
	print("Invalid source argument")

# if a video path was not supplied, grab the reference to the web cam
# if not args.get("video", False):
	# print("[INFO] starting video stream...")
	# vs = VideoStream(src=0).start()
	# time.sleep(1.0)

# # otherwise, grab a reference to the video file
# else:
	# vs = cv2.VideoCapture(args["video"])
	
	
# Box parameters
box_x = 300
global box_y
box_y = 300
box_width = 25
box_height= 25

boxSize_step = 10
boxSize_min = 10

boxPosition_step = 5

firstFrame = True




def on_press(key): 
	global box_y
	if (key == Key.up):
		box_y = box_y - boxPosition_step
	elif (key == Key.down):
		print('{0} pressed'.format(key))
	elif (key == Key.right):
		print('{0} pressed'.format(key))
	elif (key == Key.left):
		print('{0} pressed'.format(key))
	else :
		print('{0} pressed'.format(key))
	

def on_release(key):
    print('{0} release'.format(key))
    if key == Key.esc:
        # Stop listener
        return False
		
		
def clickCallback(event, x, y, flags, param):
	# grab references to the global variables
	global box_x, box_y, box_width, box_height
	# check to see if the left mouse button was released
	if event == cv2.EVENT_LBUTTONUP:
		# record the positions of x and y
		box_x = int(x - (box_width/2))
		box_y = int(y - (box_height/2))
		#print(box_x, box_y)
		#print("Click detected")

# Collect events until released (blocking)
# with Listener(
        # on_press=on_press,
        # on_release=on_release) as listener:
    # listener.join()
	
# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


fps=FPS().start()

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", clickCallback)


# loop over frames from the video stream
while True:
	time.sleep(0.03) # slow down video
	
	
	# update our box
	box_cursor = (box_x, box_y, box_width , box_height)

	if (args["source"]=="network"):
		# from https://www.youtube.com/watch?v=-mJXEzSD1Ic
		img_resp = requests.get(url)
		img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
		frame = cv2.imdecode(img_arr, -1)

	else :#handle webcam or file
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		ret, frame = vs.read()
		# Normalize frame to get decent brightness
		cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
		#frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	#frame = imutils.resize(frame, width=600)
	
	if (firstFrame):
		height,width, channels = frame.shape
		print(height, width, channels)
		box_x = int((width/2) - (box_width/2))
		box_y = int((height/2) - (box_height/2))
		firstFrame = False

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)
	#NOTE: success is not an array/list with success value for each tracker

	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		if (success == True):
			color_box = (0,255,0)#green
		else:
			color_box = (0, 0, 255)#red
		
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), color = color_box, thickness = 3)


	# Also show the Cursor Rectangle
	cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), color = (0,255,255), thickness =1)

	fps.update()


	# show the output frame
	cv2.imshow("Frame", frame)
	#key = cv2.waitKey(1) & 0xFF # doesnt work for arrow keys
	key = cv2.waitKeyEx(1) 

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if (key& 0xFF) == ord("t"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=False)

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)
		
	elif (key& 0xFF) == ord("x"):
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box_cursor)
		
	elif (key& 0xFF) == ord("c"):
		# clear trackers
		print("Clearing trackers")
		trackers.clear() # doesn't work
		trackers = cv2.MultiTracker_create() # This will cause a memory leak. But don't know other way
	
	elif (key& 0xFF) == ord("w"):
		box_height = box_height + boxSize_step
		
	elif (key& 0xFF) == ord("s"):
		box_height = box_height - boxSize_step
		box_height =  boxSize_min if (box_height < boxSize_min) else box_height
		
	elif (key& 0xFF) == ord("d"):
		box_width = box_width + boxSize_step
		
	elif (key& 0xFF) == ord("a"):
		box_width = box_width - boxSize_step
		box_width =  boxSize_min if (box_width < boxSize_min) else box_width
		
	#elif key == 2490368: 	# Up Key
		#print ("up key")
		#box_y = box_y - boxPosition_step
		#box_y = box_y
		
	elif key == 2621440:	# Down Key
		#print ("down key")
		box_y = box_y + boxPosition_step
		
	elif key == 2555904:	# Right Key
		#print ("right key")
		box_x = box_x + boxPosition_step
		
	elif key == 2424832:	# Left Key
		#print ("left key")
		box_x = box_x - boxPosition_step

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
		
	#else:
	#	print(key)
	
	
	
fps.stop()
print("FPS: ",fps.fps())
	

# if we are using a webcam, release the pointer
if not args.get("video", False):
	#vs.stop()
	print("Exiting")

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()