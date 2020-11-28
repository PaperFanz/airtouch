#
#   calib.py 10-point multitouch calibration sequence
#

import math
import time
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from queue import Queue

GREEN = (0,255,0)
BLUE = (255,0,0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

PASTEL_RED = (255, 133, 133)
PASTEL_ORANGE = (255, 200, 133)
PASTEL_YELLOW = (252, 255, 153)
PASTEL_GREEN = (170, 255, 153)
PASTEL_CYAN = (112, 243, 255)
PASTEL_BLUE = (133, 180, 255)
PASTEL_PURPLE = (168, 153, 255)
PASTEL_MAGENTA = (255, 173, 255)

PALLETE = [
    PASTEL_RED,
    PASTEL_ORANGE,
    PASTEL_YELLOW,
    PASTEL_GREEN,
    PASTEL_CYAN,
    PASTEL_BLUE,
    PASTEL_PURPLE,
    PASTEL_MAGENTA
]

DECIMATION_FACTOR = 2

DEBUG_RAW = None
DEBUG_POST = None
DEBUG_MASK = None
DEBUG_THRESH = None

def dilatation(src, size, shape):
    dil_s = size
    dil_t = shape
    kern = cv.getStructuringElement(dil_t, (2*dil_s + 1, 2*dil_s+1), (dil_s, dil_s))
    return cv.dilate(src, kern)

def erosion(src, size, shape):
    dil_s = size
    dil_t = shape
    kern = cv.getStructuringElement(dil_t, (2*dil_s + 1, 2*dil_s+1), (dil_s, dil_s))
    return cv.erode(src, kern)

class MinMaxPainter:
    def __init__(self, width, height):
        # drawing canvas
        self.canvas = np.zeros(shape=[height, width, 3], dtype=np.uint8)
        self.canvas[:,:,:] = 255 # initialize canvas to white

        # track running average of brush locations
        self.x_avg = int(width / 2)
        self.y_avg = int(height / 2)
        self.v_avg = 0

        # track if we want to start a new line
        self.track_COUNT = 4 # get 4 points before drawing a new line
        self.track = self.track_COUNT

        # depth limit
        self.depth_max = 0
        self.depth_min = 0

        # background depth constant
        self.BACKGROUND = 1000000

    def calibrate(self, depth, r):
        self.depth_min = int(depth - r/2)
        self.depth_max = int(depth + r/2)

    def paint(self, np_depth):
        np_depth[np_depth < self.depth_min] = self.BACKGROUND
        np_depth[np_depth > self.depth_max] = self.BACKGROUND

        min_v, max_v, minloc, maxloc = cv.minMaxLoc(np_depth)
        if min_v != self.BACKGROUND:
            x, y = minloc
            if self.track > 0:
                self.track = self.track - 1
                self.x_avg = int((x*2 + self.x_avg*2) / 3)
                self.y_avg = int((y*2 + self.y_avg*2) / 3)
            else:
                # do running average on locations to smooth out jitter
                x_new = int((x*2 + self.x_avg*2) / 3)
                y_new = int((y*2 + self.y_avg*2) / 3)

                self.canvas = cv.line(self.canvas, (self.x_avg, self.y_avg), (x_new, y_new), (0, 0, 0), 2)
                self.x_avg = x_new
                self.y_avg = y_new
        else:
            self.track = self.track_COUNT

    def get_canvas(self):
        return self.canvas

class ContourPainter:
    def __init__(self, width, height):
        self.h = height
        self.w = width

        # drawing canvas
        self.clear_canvas()

        # track running average of brush locations
        self.x_avg = int(self.w / 2)
        self.y_avg = int(self.h / 2)
        self.d_avg = 650
        self.v_avg = 0

        # pointer location
        self.pointer = (self.x_avg, self.y_avg)
        self.pointer_rad = 0

        # track if we want to start a new line
        self.track_COUNT = 4 # get points before drawing a new line
        self.track = self.track_COUNT

        # depth limit
        self.critical_depth = 650
        self.tolerance = 250
        self.depth_max = 400
        self.depth_min = 900
        self.min_tolerance = 0.1

        # smoothing factors
        self.al = 0.4
        self.dl = 0.1

        # background depth constant
        self.BACKGROUND = 10000

        # paint color
        self.pallete_ind = 0
        self.active_color = PALLETE[self.pallete_ind]

    def calibrate(self, depth, r, a, d, mt):
        self.critical_depth = depth
        self.tolerance = r
        self.depth_min = int(depth - r/2)
        self.depth_max = int(depth + r/2)
        self.al = a
        self.dl = d


    def hull_defects(self, cnt):
        hull = cv.convexHull(cnt, returnPoints=False)
        hullset = cv.convexityDefects(cnt, hull)
        defects = []
        if hullset is not None:
            for i in range(hullset.shape[0]):
                _,_,f,_ = hullset[i,0]
                far = tuple(cnt[f][0])
                defects.append(far)
        hull = cv.convexHull(cnt, returnPoints=True)
        return (hull, defects)


    def hull_centroid(self, hull):
        m = cv.moments(hull)
        return (int(m['m10']/m['m00']),int(m['m01']/m['m00']))


    def cursor_location(self, cnt, hull, np_depth, np_color):
        mask = np.zeros(np_depth.shape,np.uint8)
        cv.drawContours(mask,[cnt],0,255,-1)

        # dilate to make sure we don't accidentally mask out important things
        mask = dilatation(mask, 8, cv.MORPH_ELLIPSE)

        # find min location in convex hull
        min_v, _, min_loc, _ = cv.minMaxLoc(np_depth,mask = mask)
        close = ((abs(np_depth/(mask*min_v/255 + 1) - 1.0) < self.min_tolerance) * 255).astype(np.uint8)
        close = erosion(close, 2, cv.MORPH_ELLIPSE)

        global DEBUG_MASK
        DEBUG_MASK = close.copy()

        if np.count_nonzero(close) > 0:
            min_v, _, min_loc, _ = cv.minMaxLoc(np_depth,mask = close)
        else:
            min_loc = self.hull_centroid(hull)
        return min_loc, min_v


    def paint(self, np_depth, np_color):
        # debug use
        color_img = cv.resize(np_color, (0, 0), fx=0.5, fy=0.5)

        # parse out bad depth data
        np_depth[np_depth < self.depth_min] = self.BACKGROUND
        np_depth[np_depth > self.depth_max] = self.BACKGROUND
        depth = np_depth.copy()

        # reduce image scale and threshold -> this makes for a competent hand mask
        depth = np.log(depth + 1)
        depth = np.interp(depth, (depth.min(), depth.max()), (0, 255)).astype(np.uint8)
        # cv.imshow("debug", interp)
        thresh = cv.adaptiveThreshold(depth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        # cv.imshow("debug", thresh)

        global DEBUG_THRESH
        DEBUG_THRESH = thresh.copy()

        # contours on thresholded image lets us make quantitative judgements
        cnt_img, cnt, h = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

        found = False
        for i in range(len(cnt)):
            hull = cv.convexHull(cnt[i], returnPoints=True)
            a = cv.contourArea(hull)
            if a > 500 and a < 5000:
                loc, d = self.cursor_location(cnt[i], hull, np_depth, np_color)

                x, y = loc

                # img = cv.circle(color_img, loc, 5, RED, -1)
                # cv.imshow("debug", img)
                
                if self.track == self.track_COUNT:
                    # start tracking new pointer
                    self.track = self.track - 1
                    self.x_avg = x
                    self.y_avg = y
                    self.d_avg = d
                    self.v_avg = 1.0

                    # # set new color
                    # self.pallete_ind = (self.pallete_ind + 1) % len(PALLETE)
                    # self.active_color = PALLETE[self.pallete_ind]

                else:
                    x_new = 0.0
                    y_new = 0.0
                    d_new = 0.0
                    v_new = 0.0
                    
                    v = ((x - self.x_avg)**2 + (y - self.y_avg)**2)
                    if abs(v - self.v_avg) > 10.0 or abs(d - self.d_avg) > 5.0:
                        x_new = x*self.dl + self.x_avg*(1-self.dl)
                        y_new = y*self.dl + self.y_avg*(1-self.dl)
                        d_new = d*self.dl + self.d_avg*(1-self.dl)
                        v_new = (v*self.dl + self.v_avg*(1-self.dl))
                    else:
                        x_new = x*self.al + self.x_avg*(1-self.al)
                        y_new = y*self.al + self.y_avg*(1-self.al)
                        d_new = d*self.al + self.d_avg*(1-self.al)
                        v_new = (v*self.al + self.v_avg*(1-self.al))

                    if self.track > 0:
                        # wait for new pointer to stabilize
                        self.track = self.track - 1
                    elif d_new > self.critical_depth:
                        # new pointer has stabilized, but we are hovering
                        self.pointer = (int(x_new * 2), int(y_new * 2))
                        self.pointer_rad = int(np.log(d_new - self.critical_depth) * 2)
                    else:
                        # new pointer has stabilized, and we are drawing
                        self.pointer_rad = int(0)
                        thickness = int(np.log(self.critical_depth - d_new + 1))
                        pt_old = (int(self.x_avg * DECIMATION_FACTOR), int(self.y_avg * DECIMATION_FACTOR))
                        pt_new = (int(x_new * DECIMATION_FACTOR), int(y_new * DECIMATION_FACTOR))
                        self.canvas = cv.line(self.canvas, pt_old, pt_new, self.active_color, thickness)

                    self.x_avg = x_new
                    self.y_avg = y_new
                    self.d_avg = d_new
                    self.v_avg = v_new

                found = True

        if not found:
            self.track = self.track_COUNT

        return self.canvas, self.pointer, self.pointer_rad, self.active_color


    def clear_canvas(self):
        self.canvas = np.zeros(shape=[self.h, self.w, 3], dtype=np.uint8)
        self.canvas[:,:,:] = 12 # initialize canvas to white


    def get_canvas(self):
        return self.canvas

# END CONTOURPAINTER DECLARATIONS

# record mp4 of jank cajiggery drawing app
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter("out.mp4", fourcc, 20.0, (1280,1440))
out1 = cv.VideoWriter("out_raw.mp4", fourcc, 20.0, (1280,720))
out2 = cv.VideoWriter("out_post.mp4", fourcc, 20.0, (640,360))
out3 = cv.VideoWriter("out_debug.mp4", fourcc, 20.0, (640,720))

# Configure depth and color streams
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipe.start(config)

# Get stream profile and camera intrinsics
profile = pipe.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
colorizer = rs.colorizer()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, DECIMATION_FACTOR)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# application window
cv.namedWindow("Draw", cv.WINDOW_NORMAL)

# painter leaning heavily on depth sensor data to accurately parse finger point
painter = ContourPainter(1280, 720)

# depth center, depth range, smoothing alpha, smoothing delta, min tolerance
painter.calibrate(650, 250, 0.30, 0.12, 0.15)

try:
    while True:
        # grab and align frames
        frameset = pipe.wait_for_frames()
        frameset = align.process(frameset)

        # get depth  and colorframe
        depth = frameset.get_depth_frame()
        color = frameset.get_color_frame()

        np_depth = np.asanyarray(depth.get_data())
        DEBUG_RAW = np.asanyarray(colorizer.colorize(depth).get_data())

        # filter noisy noisy depth data
        depth = decimate.process(depth)
        depth = depth_to_disparity.process(depth)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        depth = disparity_to_depth.process(depth)
        # depth = hole_filling.process(depth)

        # convert to numpy arrays for cv operations
        np_depth = np.asanyarray(depth.get_data())
        np_depth = cv.flip(np_depth, 1)
        np_color = np.asanyarray(color.get_data())
        np_color = cv.flip(np_color, 1)

        DEBUG_POST = np.asanyarray(colorizer.colorize(depth).get_data())

        canvas, p, r, c = painter.paint(np_depth, np_color)

        cout = canvas.copy()
        if r > 0:
            cout = cv.circle(cout, p, r, c, 2)

        final = cv.vconcat([cout, np_color])
        cv.imshow("Draw", final)
        out.write(final)

        if DEBUG_MASK is not None:
            # debug1 = cv.vconcat([DEBUG_RAW, DEBUG_POST])
            debug2 = cv.vconcat([DEBUG_THRESH, DEBUG_MASK])
            debug2 = cv.cvtColor(debug2, cv.COLOR_GRAY2BGR)
            out1.write(DEBUG_RAW)
            out2.write(DEBUG_POST)
            out3.write(debug2)

        key = cv.waitKey(1)
        if key in (27, ord("q")):
            break

finally:
    pipe.stop()
    out.release()
    out1.release()
    out2.release()
    out3.release()
