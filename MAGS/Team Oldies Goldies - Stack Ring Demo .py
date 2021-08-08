'''
Title       : Stack Ring Guiding
Author      : JuanYi
Date        : 20210726
Hardware    : OAK-D
Depthai ver : 2.6.0
Description : This is a scaled-down version of Manual Assembly Guiding System (MAGS).
              It identifies the object to be arranged in a predefined order and prompt
              correct/wrong flag accordingly. Besides, it display the finished good
              count and the last cycle time as well as fps info.
Detail      :
            * Mobilenet SSD v2 trained with custom dataset
            * Detected object info fed to an array where only highest confidence object to pick
            * Determine if each object align in predefined order:
                * All ring within body box
                * Green above blue
                * Yellow above green
                * Orange above yellow
                * Red above orange
            * Set the OK/NG flag if condition met
            * Return status indicator, overlap frame with OK/NG color, and random text. Count up

'''
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import random

# Function that determine if each object align in predefined order
def checkOrder_countFg(det_topOnly, frame):
    global fg_count_IL, fg_count, ct_prev, ct, text1, text2
    fg_body_OK = False
    fg_blue_OK = False
    fg_green_OK = False
    fg_yellow_OK = False
    fg_orange_OK = False
    fg_red_OK = False

    blue_in_body = False
    green_in_body = False
    yellow_in_body = False
    orange_in_body = False
    red_in_body = False

    # Fetch info from array
    body_xmin = det_topOnly[1][2]
    body_ymin = det_topOnly[1][3]
    body_xmax = det_topOnly[1][4]
    body_ymax = det_topOnly[1][5]

    blue_centx = int((det_topOnly[0][2] + det_topOnly[0][4]) / 2)
    blue_centy = int((det_topOnly[0][3] + det_topOnly[0][5]) / 2)

    green_centx = int((det_topOnly[2][2] + det_topOnly[2][4]) / 2)
    green_centy = int((det_topOnly[2][3] + det_topOnly[2][5]) / 2)

    yellow_centx = int((det_topOnly[5][2] + det_topOnly[5][4]) / 2)
    yellow_centy = int((det_topOnly[5][3] + det_topOnly[5][5]) / 2)

    orange_centx = int((det_topOnly[3][2] + det_topOnly[3][4]) / 2)
    orange_centy = int((det_topOnly[3][3] + det_topOnly[3][5]) / 2)

    red_centx = int((det_topOnly[4][2] + det_topOnly[4][4]) / 2)
    red_centy = int((det_topOnly[4][3] + det_topOnly[4][5]) / 2)

    # Check if object within body box
    if body_xmin > 0 and body_xmax != body_xmin :
        fg_body_OK = True

    if blue_centx > body_xmin and blue_centx < body_xmax and blue_centy > body_ymin and blue_centy < body_ymax:
        blue_in_body = True

    if green_centx > body_xmin and green_centx < body_xmax and green_centy > body_ymin and green_centy < body_ymax:
        green_in_body = True

    if yellow_centx > body_xmin and yellow_centx < body_xmax and yellow_centy > body_ymin and yellow_centy < body_ymax:
        yellow_in_body = True

    if orange_centx > body_xmin and orange_centx < body_xmax and orange_centy > body_ymin and orange_centy < body_ymax:
        orange_in_body = True

    if red_centx > body_xmin and red_centx < body_xmax and red_centy > body_ymin and red_centy < body_ymax:
        red_in_body = True

    # Check for blue if order correct & set the flag accordingly
    if blue_in_body:
        if blue_in_body and green_in_body is False and yellow_in_body is False and orange_in_body is False and red_in_body is False:
            color_frame(0, frame, "OK")
            fg_blue_OK = True
        elif blue_in_body and green_in_body and blue_centy > green_centy:
            color_frame(0, frame, "OK")
            fg_blue_OK = True
        else:
            color_frame(0, frame, "NG")
            fg_blue_OK = False

    # Check for green if order correct & set the flag accordingly
    if green_in_body:
        if blue_in_body and green_in_body and yellow_in_body is False and orange_in_body is False and red_in_body is False:
            color_frame(2, frame, "OK")
            fg_green_OK = True
        elif blue_in_body and green_in_body and yellow_in_body and green_centy > yellow_centy:
            color_frame(2, frame, "OK")
            fg_green_OK = True
        else:
            color_frame(2, frame, "NG")
            fg_green_OK = False

    # Check for yellow if order correct & set the flag accordingly
    if yellow_in_body:
        if blue_in_body and green_in_body and yellow_in_body and orange_in_body is False and red_in_body is False:
            color_frame(5, frame, "OK")
            fg_yellow_OK = True
        elif blue_in_body and green_in_body and yellow_in_body and orange_in_body and yellow_centy > orange_centy:
            color_frame(5, frame, "OK")
            fg_yellow_OK = True
        else:
            color_frame(5, frame, "NG")
            fg_yellow_OK = False

    # Check for orange if order correct & set the flag accordingly
    if orange_in_body:
        if blue_in_body and green_in_body and yellow_in_body and orange_in_body and red_in_body is False:
            color_frame(3, frame, "OK")
            fg_orange_OK = True
        elif blue_in_body and green_in_body and yellow_in_body and orange_in_body and red_in_body and orange_centy > red_centy:
            color_frame(3, frame, "OK")
            fg_orange_OK = True
        else:
            color_frame(3, frame, "NG")
            fg_orange_OK = False

    # Check for red if order correct & set the flag accordingly
    if red_in_body:
        if blue_in_body and green_in_body and yellow_in_body and orange_in_body and red_in_body and orange_centy > red_centy:
            color_frame(4, frame, "OK")
            fg_red_OK = True
        else:
            color_frame(4, frame, "NG")
            fg_red_OK = False

    # Light up the legend box if each condition met
    legendX = 680
    legendY = 710
    cv2.putText(frame, "Body", (legendX - 65, legendY - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 20, legendY - 20), (legendX + 20, legendY), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_body_OK:
        cv2.rectangle(frame, (legendX - 20, legendY - 20), (legendX + 20, legendY), (100, 255, 100), thickness=-1)

    cv2.putText(frame, "Blue", (legendX - 65, legendY - 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 16, legendY - 50), (legendX + 16, legendY - 30), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_blue_OK:
        cv2.rectangle(frame, (legendX - 16, legendY - 50), (legendX + 16, legendY - 30), (100, 255, 100), thickness=-1)

    cv2.putText(frame, "Green", (legendX - 65, legendY - 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 12, legendY - 80), (legendX + 12, legendY - 60), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_green_OK:
        cv2.rectangle(frame, (legendX - 12, legendY - 80), (legendX + 12, legendY - 60), (100, 255, 100), thickness=-1)

    cv2.putText(frame, "Yellow", (legendX - 65, legendY - 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 8, legendY - 110), (legendX + 8, legendY - 90), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_yellow_OK:
        cv2.rectangle(frame, (legendX - 8, legendY - 110), (legendX + 8, legendY - 90), (100, 255, 100), thickness=-1)

    cv2.putText(frame, "Orange", (legendX - 65, legendY - 125), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 6, legendY - 140), (legendX + 6, legendY - 120), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_orange_OK:
        cv2.rectangle(frame, (legendX - 6, legendY - 140), (legendX + 6, legendY - 120), (100, 255, 100), thickness=-1)

    cv2.putText(frame, "Red", (legendX - 65, legendY - 155), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
    cv2.rectangle(frame, (legendX - 4, legendY - 170), (legendX + 4, legendY - 150), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX)
    if fg_red_OK:
        cv2.rectangle(frame, (legendX - 4, legendY - 170), (legendX + 4, legendY - 150), (100, 255, 100), thickness=-1)

    # Count how many label id with confidence != 0
    det_size = 0
    for i in range(6):
        if det_topOnly[i][1] != 0:
            det_size += 1
            print(f"det_size: {det_size}")

    # Check if all condition mets and ready to count up
    if det_size == 6 and fg_blue_OK and fg_green_OK and fg_yellow_OK and fg_orange_OK and fg_red_OK and fg_count_IL is False:
        fg_count_IL = True
        fg_count += 1
        print(f"FG counter: {fg_count}")

        ct_current = time.monotonic()
        ct = (ct_current - ct_prev)
        ct_prev = ct_current

        rand_text1 = random.randint(0, 3)
        rand_text2 = random.randint(0, 3)
        if rand_text1 == 0:
            text1 = "Well done!"
        elif rand_text1 == 1:
            text1 = "Good job!"
        elif rand_text1 == 2:
            text1 = "Awesome!"
        elif rand_text1 == 3:
            text1 = "Amazing!"

        if rand_text2 == 0:
            text2 = "Keep it up."
        elif rand_text2 == 1:
            text2 = "Keep going."
        elif rand_text2 == 2:
            text2 = "Let's strike again."
        elif rand_text2 == 3:
            text2 = "You're the best"

    elif det_size == 0 and fg_count_IL is True:
        fg_count_IL = False

    if fg_count_IL:
        print(f"{text1} {text2}")
        cv2.putText(frame, f"{text1} {text2}", (160, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0))
        cv2.putText(frame, f"{text1} {text2}", (161, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,50,0))
        cv2.putText(frame, f"{text1} {text2}", (160, 101), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,50,0))
        cv2.putText(frame, f"{text1} {text2}", (161, 101), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0))

    # Reset all position
    for i in range(6):
        for j in range(6):
            det_topOnly[j][i] = 0
    return det_topOnly, frame

# Color the subframe based on OK/NG flag
def color_frame(key, frame, flag):
    sub_x1 = det_topOnly[key][2]
    sub_y1 = det_topOnly[key][3]
    sub_x2 = det_topOnly[key][4]
    sub_y2 = det_topOnly[key][5]
    sub_frame = frame[sub_y1:sub_y2, sub_x1:sub_x2]
    color_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 0

    if flag == "OK":
        # Setting blue/green/red to desired value
        color_rect[:, :, 0] = 100
        color_rect[:, :, 1] = 250
        color_rect[:, :, 2] = 100

    elif flag == "NG":
        color_rect[:, :, 0] = 100
        color_rect[:, :, 1] = 100
        color_rect[:, :, 2] = 250

    result_frame = cv2.addWeighted(sub_frame, 0.8, color_rect, 0.4, 1.0)
    # Putting the image back to its position
    frame[sub_y1:sub_y2, sub_x1:sub_x2] = result_frame
    return frame

# Crop the frame to rectangle
def crop_to_rect(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height)/2)
    return frame[0: height, delta:width - delta]

# Blob path
nnBlobPath = str((Path(__file__).parent / Path('models/mobilenet-ssd-openvino_2012.3_6shave_StackRingR4.blob')).resolve().absolute())

# MobilenetSSD label texts
labelMap = ["unknown", "Blue", "Body", "Green", "Orange", "Red", "Yellow"]

print("Create pipeline...")
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

# Setup color camera
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(2, 3)   # to match the monocamera 720p
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)
cam.initialControl.setManualExposure(20000, 600)

# color camera setting for spatial neural network
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Setup mono camera
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setup stereo
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(245)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_3x3)
#stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(False)
#stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

print("Create spatial neural network ")
sdn = pipeline.createMobileNetSpatialDetectionNetwork()
sdn.setBlobPath(nnBlobPath)
sdn.setConfidenceThreshold(0.5)
sdn.input.setBlocking(False)
sdn.setBoundingBoxScaleFactor(0.2)
sdn.setDepthLowerThreshold(100)
sdn.setDepthUpperThreshold(5000)

cam.preview.link(sdn.input)
stereo.depth.link(sdn.inputDepth)

# Setup stream
cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("cam")
cam.isp.link(cam_xout.input)

depth_xout = pipeline.createXLinkOut()
depth_xout.setStreamName("dep")
sdn.passthroughDepth.link(depth_xout.input)

sdn_xout = pipeline.createXLinkOut()
sdn_xout.setStreamName("det")
sdn.out.link(sdn_xout.input)

bbox_xout = pipeline.createXLinkOut()
bbox_xout.setStreamName("bbox")
sdn.boundingBoxMapping.link(bbox_xout.input)

print("Pipeline created.")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    camQ = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    depQ = device.getOutputQueue(name="dep", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    bboxQ = device.getOutputQueue(name="bbox", maxSize=4, blocking=False)

    # For fps counting
    startTime = time.monotonic()
    counter = 0
    fps = 0

    # For finished good counting
    fg_count_IL = False
    fg_count = 0

    # For cycle time counting
    ct_prev = time.monotonic()
    ct = 0

    text1 = ""
    text2 = ""

    det_topOnly = [[0 for i in range(6)] for j in range(7)]

    while True:
        inCam = camQ.get()
        inDet = detQ.get()
        inDep = depQ.get()

        counter += 1
        current_time = time.monotonic()

        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = crop_to_rect(inCam.getCvFrame())
        depthFrame = inDep.getFrame()

        height = frame.shape[0]
        width = frame.shape[1]

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        depthHeight = depthFrameColor.shape[0]
        depthWidth = depthFrameColor.shape[1]
        depthDelta = int((depthWidth - depthHeight) / 2)

        detections = inDet.detections
        if len(detections) != 0:
            
            bboxMap = bboxQ.get()
            bboxes = bboxMap.getConfigData()

            for bbox in bboxes:
                roi = bbox.roi
                roi = roi.denormalize(int(depthFrameColor.shape[1]), depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0, 255, 0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # To feed detected object to new array
            det_temp = [[0 for i in range(6)] for j in range(len(detections))]  # Create empty array based on how many being detected

            i = 0
            for detected in detections:     # Fetch info from each detected into column
                det_temp[i][0] = detected.label
                det_temp[i][1] = int(round(detected.confidence * 100,0))
                det_temp[i][2] = int(detected.xmin * width)
                det_temp[i][3] = int(detected.ymin * height)
                det_temp[i][4] = int(detected.xmax * width)
                det_temp[i][5] = int(detected.ymax * height)
                i += 1

            # Filter the row based on the object ID
            det_tempArr = np.array(det_temp)
            det_filter1 = det_tempArr[det_tempArr[:, 0] == 1]
            det_filter2 = det_tempArr[det_tempArr[:, 0] == 2]
            det_filter3 = det_tempArr[det_tempArr[:, 0] == 3]
            det_filter4 = det_tempArr[det_tempArr[:, 0] == 4]
            det_filter5 = det_tempArr[det_tempArr[:, 0] == 5]
            det_filter6 = det_tempArr[det_tempArr[:, 0] == 6]

            # Create empty array before retrieve in the top row (highest confidence)
            det_top1 = [0 for i in range(6)]
            det_top2 = [0 for i in range(6)]
            det_top3 = [0 for i in range(6)]
            det_top4 = [0 for i in range(6)]
            det_top5 = [0 for i in range(6)]
            det_top6 = [0 for i in range(6)]

            # Retrieve only the top row
            if len(det_filter1) != 0:
                det_top1 = det_filter1[0, :]
            if len(det_filter2) != 0:
                det_top2 = det_filter2[0, :]
            if len(det_filter3) != 0:
                det_top3 = det_filter3[0, :]
            if len(det_filter4) != 0:
                det_top4 = det_filter4[0, :]
            if len(det_filter5) != 0:
                det_top5 = det_filter5[0, :]
            if len(det_filter6) != 0:
                det_top6 = det_filter6[0, :]

            # Stack all the top row into one single array
            det_topOnly = np.stack((det_top1, det_top2, det_top3, det_top4, det_top5, det_top6), axis=0)

            # Draw bbox, label & confidence on frame
            for i in range(6):
                if det_topOnly[i][1] != 0:  # Check confidence if not zero
                    label = labelMap[int(det_topOnly[i][0])]
                    confidence = det_topOnly[i][1]
                    x1 = det_topOnly[i][2]
                    y1 = det_topOnly[i][3]
                    x2 = det_topOnly[i][4]
                    y2 = det_topOnly[i][5]
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "{:.0f}%".format(confidence), (x1 - 50, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

        # Crop the depth frame after draw bounding box mapping
        depthFrameColorCrop = crop_to_rect(depthFrameColor)

        # To check all the position of detected if in correct order
        checkOrder_countFg(det_topOnly, frame)

        cv2.putText(frame, f"FG Count: {fg_count}", (5, frame.shape[0] - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(frame, "Last CT  : {:.2f}s".format(ct), (5, frame.shape[0] - 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.putText(frame, "NN fps   : {:.2f}".format(fps), (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
        cv2.imshow("depth", depthFrameColorCrop)
        cv2.imshow("cam", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("reset FG counter.")
            fg_count = 0
            ct_prev = time.monotonic()