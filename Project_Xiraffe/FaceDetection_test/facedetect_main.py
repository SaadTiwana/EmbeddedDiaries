'''
Copyright 2020 Avnet Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# This code is modified from code here: 

# USAGE
# python avnet_face_detection.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35]

import numpy as np
import argparse
import imutils
import time
import cv2
import os, errno

from imutils.video import FPS

import xir.graph
import pathlib
import xir.subgraph

import runner
#from vitis_ai_vart.facedetect import FaceDetect
import facedetect



def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children
            if s.metadata.get_attr_str ("device") == "DPU"]
    return sub


#pipeline_out = "appsrc ! queue ! autovideosink sync=false"
pipeline_out = "appsrc ! queue ! fpsdisplaysink video-sink='autovideosink' sync=false text-overlay=true fullscreen-overlay=1"

out = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, 0, 20.0, (1920,1080))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help = "input camera identifier (default = 0)")
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
args = vars(ap.parse_args())

if not args.get("input",False):
  inputId = 0
else:
  inputId = int(args["input"])
print('[INFO] input camera identifier = ',inputId)

if not args.get("detthreshold",False):
  detThreshold = 0.55
else:
  detThreshold = float(args["detthreshold"])
print('[INFO] face detector - softmax threshold = ',detThreshold)

if not args.get("nmsthreshold",False):
  nmsThreshold = 0.35
else:
  nmsThreshold = float(args["nmsthreshold"])
print('[INFO] face detector - NMS threshold = ',nmsThreshold)

# Initialize Vitis-AI/DPU based face detector
#dpu = runner.Runner("/usr/share/vitis_ai_library/models/densebox_640_360")[0]

# Using this instead
model='densebox_640_360.elf'
path=pathlib.Path(model)
print(path)
g = xir.graph.Graph.deserialize(path)
subgraphs = get_subgraph(g)
print("hello")
print(subgraphs)
print(len(subgraphs))
assert len(subgraphs) == 1 # only one DPU kernel
dpu=runner.Runner(subgraphs[0], "run")



dpu_face_detector = facedetect.FaceDetect(dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

# Initialize the camera input
print("[INFO] starting camera input ...")
cam = cv2.VideoCapture(inputId)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not (cam.isOpened()):
    print("[ERROR] Failed to open camera ", inputId )
    exit()

# start the FPS counter
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# Capture image from camera
	ret,frame = cam.read()

	# Vitis-AI/DPU based face detector
	faces = dpu_face_detector.process(frame)

	# loop over the faces
	for i,(left,top,right,bottom) in enumerate(faces): 

		# draw a bounding box surrounding the object so we can
		# visualize it
		cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)


	# Display the processed image
	#cv2.imshow("Face Detection", frame)
	frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_AREA)
	out.write(frame) 
	key = cv2.waitKey(1) & 0xFF

	# Update the FPS counter
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Stop the face detector
dpu_face_detector.stop()
del dpu

# Cleanup
cv2.destroyAllWindows()
