#
#   calib.py 10-point multitouch calibration sequence
#

import math
import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from queue import Queue

print("Environment Loaded")

# Configure depth and color streams
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipe.start(config)

print("Stream started")

# Get stream profile and camera intrinsics
profile = pipe.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# # frame FIFO for running avg & temporal smoothing
# fq = Queue(maxsize=10)

# # populate frame fifo
# for x in range(10):
#     frameset = pipe.wait_for_frames()
#     depth_frame = frameset.get_depth_frame()
#     fq.put(depth_frame)

def stretch(c):
    # c = np.log(c.astype('uint16') + 1) # log before stretch?
    min_v = np.percentile(c, 5)
    max_v = np.percentile(c, 95)
    P = (255)/(max_v - min_v)
    img = (c -  min_v).astype('float') * P
    img = np.clip(img, 0, 255).astype('uint8')
    return img

try:
    while True:
        # Grab some frames
        frameset = pipe.wait_for_frames()
        frame = frameset.get_depth_frame()

        frame = decimate.process(frame)
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        frame = disparity_to_depth.process(frame)
        frame = hole_filling.process(frame)
        filtered_depth = np.asanyarray(frame.get_data())[2:353,50:]

        rgb_depth = cv.cvtColor(stretch(filtered_depth), cv.COLOR_GRAY2RGB)
        min_v, max_v, minloc, maxloc = cv.minMaxLoc(filtered_depth)
        rgb_depth = cv.circle(rgb_depth, minloc, 20, (0, 0, 255), thickness=5)

        cv.imshow("Processed Depth Colorized", rgb_depth)
        key = cv.waitKey(1)

        if key in (27, ord("q")):
            break

finally:
    pipe.stop()
