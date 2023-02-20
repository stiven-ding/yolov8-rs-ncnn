import argparse
import time
from pathlib import Path

import cv2

from numpy import random


import pyrealsense2 as rs
import numpy as np
#################################

ENABLE_YOLO_DETECT = True

#################################

if ENABLE_YOLO_DETECT:
    from ncnn.utils import draw_detection_objects
    from yolov8 import YoloV8s


def sample_distance(depth_image, mid_x, mid_y):
    global depth_scale
    window = 2

    sample_depth = depth_image[mid_y-window:mid_y+window, mid_x-window:mid_x+window].astype(float)
    dist, _, _, _ = cv2.mean(sample_depth)
    dist = dist * depth_scale

    return dist

def draw_label(image, text, x0, y0, x1, y1):

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
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        image,
        text,
        (int(x), int(y + label_size[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
    )


def draw_detection_objects(image, depth_image, class_names, objects, min_prob=0.0):
    for obj in objects:
        if obj.prob < min_prob:
            continue

        #print(
        #    "%d = %.5f at %.2f %.2f %.2f x %.2f"
        #    % (obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        #)

        x0, y0, x1, y1 = obj.rect.x, obj.rect.y, obj.rect.x+obj.rect.w, obj.rect.y+obj.rect.h
        mid_x, mid_y = round(int(x0+x1) /2), round(int(y0 + y1)/2)

        dist = sample_distance(depth_image, mid_x, mid_y)
        name = class_names[int(obj.label)]
        conf = obj.prob * 100

        text = f'{name} {dist:.2f}m {conf:.0f}%'

        draw_label(image, text, x0, y0, x1, y1)

def detect(save_img=False):

    global device, model, half, depth_scale
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Initialize YOLO
    if ENABLE_YOLO_DETECT:
        print('Loading model')
        net = YoloV8s(
            target_size=412,
            prob_threshold=0.25,
            nms_threshold=0.45,
            num_threads=4,
            use_gpu=True,
        )
        

    # Reset camera, bug workaround. Min distance invalid when used
    #print("RS reset start")
    #ctx = rs.context()
    #devices = ctx.query_devices()
    #for dev in devices:
    #    dev.hardware_reset()
    #print("RS reset done")
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

    while True:
        #t0 = time.time()
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
        #depth_to_disparity = rs.disparity_transform(True)
        #disparity_to_depth = rs.disparity_transform(False)
        spatial = rs.spatial_filter(smooth_alpha=1,smooth_delta=50,magnitude=5,hole_fill=3)
        hole_filling = rs.hole_filling_filter(1)

        #depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        #depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        img = np.asanyarray(color_frame.get_data())
        im0 = img.copy()

        depth_img = np.asanyarray(depth_frame.get_data())
        invalid = np.full((480,640),255, dtype=np.uint8)
        depth_img = np.where(depth_img[:,:] == [0,0], invalid, depth_img)

        #depth_img = cv2.bilateralFilter((depth_img/256.0).astype(np.uint8), 9, 75, 75)
        depth_img = cv2.medianBlur(depth_img,5)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # Depth range selection
        #depth_l1_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l1, alpha=0.2), cv2.COLORMAP_JET)
        #depth_l2 = cv2.inRange(depth_image,0.5/depth_scale, 1.0/depth_scale)
        #depth_l2_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_l2, alpha=0.2), cv2.COLORMAP_JET)
        #contours_l2, _ = cv2.findContours(depth_l2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #

        # Process contours
        #edged = cv2.Canny(depth_img.astype(np.uint8), 50, 200)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        #dilate = cv2.dilate(edged, kernel, iterations =1)
        #contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(depth_colormap, contours, -1, (0,255,0), 2)

        contours = []
        c_start, c_step, c_levels = 0.0, 0.5, 3
        for i in range(c_levels):
            depth_range = cv2.inRange(depth_img,c_start/depth_scale, (c_start+c_step)/depth_scale)
            contours_range, _ = cv2.findContours(depth_range,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_range:
                contours.append(c)
            #cv2.imshow("Result depth" + str(c_start) + " " + str(c_start+c_step),cv2.cvtColor(depth_range, cv2.COLOR_BGR2RGB))
            c_start += c_step

        for c in contours:
            #cv2.convexHull(c)
            size = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)

            if w > 50 and h > 50 and w < 640 and h < 480:
                cv2.rectangle(depth_colormap, (x,y), (x+w,y+h), (0,200,100),2)
                M = cv2.moments(c)
                mid_x, mid_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

                dist = sample_distance(depth_img, mid_x, mid_y)
                cv2.putText(depth_colormap, "dist: " + str(round(dist,2)) + "m", 
                (x, y-7),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,200,0),2)

        if ENABLE_YOLO_DETECT:
            # Inference
            #t1 = time_synchronized()
            objects = net(img)
            #t2 = time_synchronized()

            # Process detections
            draw_detection_objects(im0, depth_img, net.class_names, objects)

            # Print time (inference + NMS)
            #print(f' ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        cv2.imshow("YOLOv7 result", im0)
        #cv2.imshow("Depth L1 result",cv2.cvtColor(depth_l1_colormap, cv2.COLOR_BGR2RGB))
        cv2.imshow("L result depth",cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    print(opt)

    detect()

