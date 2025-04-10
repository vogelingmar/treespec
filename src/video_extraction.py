#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test trained network on a video
"""
from __future__ import absolute_import
import os
import cv2
import torch
import ffmpeg

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.logger import setup_logger

setup_logger()


#  model and video variables
MODEL_NAME = "X-101_RGB_60k.pth"
VIDEO_PATH = "/data/training data/Sauen_Mapping_Dataset/20fps_Sauen.mp4"
# corrected_video_path = '/home/ingmar/Documents/repos/treespec/src/io/videos/corrected_video.mp4'
CORRECTED_VIDEO_PATH = "/data/training data/Sauen_Mapping_Dataset/cropped_20fps_Sauen.mp4"
BARK_DIR = "/home/ingmar/Documents/repos/treespec/src/io/pictures/"

if __name__ == "__main__":
    torch.cuda.is_available()
    logger = setup_logger(name=__name__)

    # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True

    cfg.OUTPUT_DIR = "/home/ingmar/Documents/repos/treespec/src/io/models/"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, MODEL_NAME)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time

    # set detector
    predictor_synth = DefaultPredictor(cfg)

    # set metadata
    tree_metadata = MetadataCatalog.get("my_tree_dataset").set(
        thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"]
    )

    # ffmpeg.input(video_path).output(corrected_video_path, vf='lenscorrection=k1=-0.5:k2=-0.5,
    # crop=in_w/3:in_h/3:in_w/3:in_h/3, transpose=1').run()

    # Get one video frame
    vcap = cv2.VideoCapture(CORRECTED_VIDEO_PATH)

    # get vcap property
    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VIDEO recorder
    # Grab the stats from image1 to use for the resultant video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("/io/videos/pred_and_track_00.mp4", fourcc, 5, (w, h))

    # Check if camera opened successfully
    if vcap.isOpened() is False:
        print("Error opening video stream or file")

    vid_vis = VideoVisualizer(metadata=tree_metadata)

    nframes = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        y = 000
        # h = 800
        x = 000
        # w = 800
        crop_frame = frame[y : y + h, x : x + w]
        # cv2.imshow('frame'<, crop_frame)
        if cv2.waitKey(1) == ord("q"):
            break

        # 5 fps
        if nframes % 12 == 0:
            outputs_pred = predictor_synth(crop_frame)

            pred_tree_boxes = outputs_pred["instances"].pred_boxes.tensor.cpu().numpy()

            i = 0
            for tree_box in enumerate(pred_tree_boxes):
                box_coords = tree_box[1]  # takes coordinates of the box
                x1, y1, x2, y2 = map(int, box_coords)  # Convert to integer coordinates
                # Crop the frame based on the bounding box coordinates
                cropped_box = crop_frame[y1:y2, x1:x2]

                if (x2 - x1) * (y2 - y1) >= 200000:
                    angle = (480 - ((x1 + x2) / 2)) / 32  # approximates angle of tree to the camera ortientation
                    screenshot_filename = f"bark_{nframes:04d}_box_{i:02d}_angle_{angle:.2f}.jpg"
                    # Name screenshot by frame index and box index
                    screenshot_path = os.path.join(BARK_DIR, screenshot_filename)
                    cv2.imwrite(screenshot_path, cropped_box)  # Save the cropped box
                    i += 1

            out = vid_vis.draw_instance_predictions(crop_frame, outputs_pred["instances"].to("cpu"))

            vid_frame = out.get_image()
            video.write(vid_frame)
            cv2.imshow("frame", vid_frame)

        nframes += 1

    video.release()
    vcap.release()
    cv2.destroyAllWindows()
