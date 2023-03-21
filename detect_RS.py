import argparse

import time
from pathlib import Path

import ctypes

import otsu

import cv2
from numpy import random

import pyrealsense2 as rs
import numpy as np
from multiprocessing.connection import Client

#from skimage.filters import threshold_multiotsu

#################################

# DEBUG CONFIGURATIONS

ENABLE_YOLO_DETECT = False
ENABLE_AUDIO = False
ENABLE_SHOW_LAYERS = False

#################################

# X11 multithread support
ctypes.CDLL('libX11.so.6').XInitThreads()

if ENABLE_YOLO_DETECT:
    from ncnn.utils import draw_detection_objects
    from yolov8 import YoloV8s
    
class Obstable():
    def __init__(self, dist, xyxy, name, prob, mid_xy):
        self.dist = dist
        self.xyxy = xyxy
        self.name = name
        self.prob = prob
        self.mid_xy = mid_xy
  
    def __lt__(self, obj):
        return ((self.dist) < (obj.dist))
  
    def __gt__(self, obj):
        return ((self.dist) > (obj.dist))
  
    def __le__(self, obj):
        return ((self.dist) <= (obj.dist))
  
    def __ge__(self, obj):
        return ((self.dist) >= (obj.dist))
  
    def __eq__(self, obj):
        return (self.dist == obj.dist)

    def __repr__(self):
        return '{' + str(self.dist) + ', ' + self.dist + ', ' + self.prob + '}'

def ipc_connect():
    print('Starting IPC')
    address = ('localhost', 55777)
    while True:
        try:
            conn = Client(address, authkey=b'secret password')
        except:
            print('IPC trying to connect')
            time.sleep(1)
        else:
            print('IPC connected')
            return conn
            break


def sample_distance(depth_image, mid_x, mid_y):
    global depth_scale
    window = 2

    sample_depth = depth_image[mid_y-window:mid_y+window, mid_x-window:mid_x+window].astype(float)
    dist, _, _, _ = cv2.mean(sample_depth)
    dist = dist * depth_scale

    return dist

def draw_label(image, text, x0, y0, x1, y1, color=(0,0,0)):

    cv2.rectangle(
        image,
        (int(x0), int(y0)),
        (int(x1), int(y1)),
        (255, 0, 0),
    )
    
    label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    x = x0
    y = y0 - label_size[1] - baseLine
    if y < 0:
        y = 0
    if x + label_size[0] > image.shape[1]:
        x = image.shape[1] - label_size[0]

    cv2.rectangle(
        image,
        (int(x), int(y)),
        (int(x + label_size[0]), int(y + label_size[1] + baseLine)),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (int(x), int(y + label_size[1])),
        cv2.LINE_AA,
        0.5,
        (255, 255, 255),
    )

def draw_detection_objects(image, depth_colormap, objects):
    for idx, obj in enumerate(objects):
        x0, y0, x1, y1 = obj.xyxy[0], obj.xyxy[1], obj.xyxy[2], obj.xyxy[3]
        text = f'{obj.name} {obj.dist:.2f}m'

        if idx < 2:
            color = (50,50,150)
        else:
            color = (0,150,100) if obj.prob < 0 else (0,0,0)

        draw_label(image, text, x0, y0, x1, y1, color)
        draw_label(depth_colormap, text, x0, y0, x1, y1, color)

def detect(save_img=False):

    t0 = time.perf_counter()

    global device, model, half, depth_scale
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Initialize YOLO
    if ENABLE_YOLO_DETECT:
        print('Loading model')
        net = YoloV8s(
            target_size=32*6,
            prob_threshold=0.25,
            nms_threshold=0.45,
            num_threads=2,
            use_gpu=True,
        )

    # Reset camera, bug workaround. Min distance invalid when used
    #print("RS reset start")
    #ctx = rs.context()
    #devices = ctx.query_devices()
    #for dev in devices:
    #    dev.hardware_reset()
    #print("RS reset done")

    if ENABLE_AUDIO:
        conn = ipc_connect()

    print('Enabling RS camera')
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor_dep = profile.get_device().first_depth_sensor()
    sensor_dep.set_option(rs.option.min_distance, 100)
    #sensor_dep.set_option(rs.option.enable_max_usable_range, 1)
    sensor_dep.set_option(rs.option.laser_power, 100)
    sensor_dep.set_option(rs.option.receiver_gain, 18)
    sensor_dep.set_option(rs.option.confidence_threshold, 1)
    sensor_dep.set_option(rs.option.noise_filtering, 2)

    depth_scale = sensor_dep.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    t1 = time.perf_counter()
    print(f' ({(1E3 * (t1 - t0)):.1f}ms) boot\n')

    while True:
        t0 = time.perf_counter()

        while True:
            try:
                frames = pipeline.wait_for_frames()
            except:
                print('Cam recv no frames')
            else:
                break

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        # filtering
        t1 = time.perf_counter()

        #depth_to_disparity = rs.disparity_transform(True)
        #disparity_to_depth = rs.disparity_transform(False)
        spatial = rs.spatial_filter(smooth_alpha=0.5,smooth_delta=30,magnitude=1,hole_fill=5)
        hole_filling = rs.hole_filling_filter(1)

        #depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        #depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        img = np.asanyarray(color_frame.get_data())
        im0 = img.copy()

        depth_img = np.asanyarray(depth_frame.get_data())
        invalid = np.full((480,640),65536, dtype=np.uint16)
        depth_img = np.where(depth_img[:,:] == [0,0], invalid, depth_img)

        #depth_img = cv2.bilateralFilter((depth_img/256.0).astype(np.uint8), 9, 75, 75)
        depth_img = cv2.medianBlur(depth_img,5)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # All obstacles list
        obstacles = []


        # Contour Detection
        t2 = time.perf_counter()
        contours = []

        # Modified Two-Stage Multi-Otsu Threasholding
        otsu_img = (depth_img // 100).astype(np.uint8).clip(0,255)
        hist = cv2.calcHist(
            [otsu_img],
            channels=[0],
            mask=None,
            histSize=[256],
            ranges=[0, 256]
        )
        try:
            thresholds = sorted(otsu.modified_TSMO(hist, M=64, L=256))
            if len(thresholds) < 1:
                raise Exception
            
            thresh_a, layer = 0, 1
            for thresh_b in thresholds:
                depth_range = cv2.inRange(otsu_img, thresh_a, thresh_b)
                h,w=depth_range.shape[0:2]
                cv2.rectangle(depth_range,(0,0),(w,h),(0),20) # really thick white rectangle
                contours_range, _ = cv2.findContours(depth_range,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours_range:
                    contours.append(c)
                if ENABLE_SHOW_LAYERS:
                    cv2.imshow("Result depth L" + str(layer), cv2.cvtColor(depth_range, cv2.COLOR_BGR2RGB))
                if layer > 1:
                    break
                thresh_a = thresh_b
                layer = layer + 1
        except:
            # Linear Thresholding for Nackup
            c_start, c_step, c_levels = 0.0, 0.5, 5
            for i in range(c_levels):
                depth_range = cv2.inRange(depth_img,c_start/depth_scale, (c_start+c_step)/depth_scale)
                h,w=depth_range.shape[0:2]
                cv2.rectangle(depth_range,(0,0),(w,h),(0),20) # really thick white rectangle
                contours_range, _ = cv2.findContours(depth_range,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours_range:
                    contours.append(c)

                if ENABLE_SHOW_LAYERS and layer < 3:
                    cv2.imshow("Result depth L" + str(layer), cv2.cvtColor(depth_range, cv2.COLOR_BGR2RGB))
                c_start += c_step
                layer = layer + 1

        # Process Contours
        for c in contours:
            #cv2.convexHull(c)
            size = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)

            if w < 50 or h < 50 or w >= 640 or h >= 480:
                continue
        
            M = cv2.moments(c)
            mid_x, mid_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            dist = sample_distance(depth_img, mid_x, mid_y)

            if dist > 0:
                obs = Obstable(dist, [x, y, x+w, y+h], '', -1, [mid_x, mid_y])
                obstacles.append(obs)


        # YOLO inference
        t3 = time.perf_counter()
        if ENABLE_YOLO_DETECT:
            objects = net(img)
            
            for obj in objects:
                if obj.prob < 0.1:
                    continue

                xyxy = [obj.rect.x, obj.rect.y, obj.rect.x+obj.rect.w, obj.rect.y+obj.rect.h]
                mid_x, mid_y = round(int(xyxy[0]+xyxy[2]) /2), round(int(xyxy[1] + xyxy[3])/2)

                dist = sample_distance(depth_img, mid_x, mid_y)
                name = net.class_names[int(obj.label)]

                if dist > 0:
                    obs = Obstable(dist, xyxy, name, obj.prob, [mid_x, mid_y])
                    obstacles.append(obs)

        # Process detections
        obstacles.sort()
        draw_detection_objects(im0, depth_colormap, obstacles)

        t4 = time.perf_counter()
        
        # Print time (inference + NMS)
        print(f' ({(1E3 * (t1 - t0)):.1f}ms) input, ({(1E3 * (t2 - t1)):.1f}ms) filter, ({(1E3 * (t3 - t2)):.1f}ms) cont, ({(1E3 * (t4 - t3)):.1f}ms) yolo, ({(1E3 * (t4 - t0)):.1f}ms) total')


        # Stream results
        if ENABLE_AUDIO:      
            if len(obstacles) > 0:
                msg = str(obstacles[0].dist) + ',' + str(obstacles[0].mid_xy[0]) + ',' + str(obstacles[0].mid_xy[1]) + ',' + obstacles[0].name
            else:
                msg = '10,0,0,'

            try:
                conn.send(msg)
            except:
                conn = ipc_connect()

        cv2.imshow("YOLOv8 result", im0)
        cv2.imshow("Depth result", cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    print(opt)

    detect()

