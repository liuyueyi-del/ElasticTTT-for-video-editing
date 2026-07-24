# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def estimate_motion(video_path: str):
    # Load the YOLOv8 model.
    model = YOLO("yolov8n.pt")
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])
    frame_count = 0
    flow_img_size = None
    frame_shape = None
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_shape = frame.shape[:2]
            # Run YOLOv8 tracking on the frame, persisting tracks between frames.
            results = model.track(frame, persist=True)
            boxes = results[0].boxes
            if boxes is None or boxes.xywh is None or len(boxes) == 0:
                frame_count += 1
                continue
            xywh = boxes.xywh
            if xywh.numel() == 0:
                frame_count += 1
                continue
            annotated_frame = results[0].plot()
            flow_img_size = annotated_frame.shape[:2]
            ids = boxes.id
            if ids is None:
                xywh_np = xywh.cpu().numpy()
                if xywh_np.ndim == 1:
                    xywh_np = xywh_np.reshape(1, -1)
                areas = xywh_np[:, 2] * xywh_np[:, 3]
                box = xywh_np[int(areas.argmax())]
                track_history[0].append([frame_count, *box])
            else:
                track_ids = ids.int().cpu().tolist()
                xywh_np = xywh.cpu().numpy()
                for box, track_id in zip(xywh_np, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append([frame_count, x, y, w, h])
            frame_count += 1
        else:
            break
    cap.release()
    if flow_img_size is None and frame_shape is not None:
        flow_img_size = frame_shape
    ret_tracks = []
    for key, items in track_history.items():
        if not items:
            continue
        frame_map = {int(item[0]): item[1:] for item in items}
        min_frame = min(frame_map)
        first_box = frame_map[min_frame]
        last_box = first_box
        full = []
        for f in range(0, min_frame):
            full.append([f, *first_box])
        for f in range(min_frame, frame_count):
            if f in frame_map:
                last_box = frame_map[f]
            full.append([f, *last_box])
        if len(full) == frame_count:
            ret_tracks.append(np.asarray(full).astype(int))
    if not ret_tracks and frame_count > 0:
        if flow_img_size is None:
            raise ValueError("No valid motion tracks detected and frame size unknown.")
        h, w = flow_img_size
        x = w / 2.0
        y = h / 2.0
        bw = w * 0.5
        bh = h * 0.5
        fallback = [[f, x, y, bw, bh] for f in range(frame_count)]
        ret_tracks.append(np.asarray(fallback).astype(int))
    return ret_tracks, flow_img_size
