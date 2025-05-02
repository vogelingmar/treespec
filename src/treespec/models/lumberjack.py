"""Lumberjack automatically extracts tree images from a video"""

from __future__ import absolute_import
from typing import Optional
import re
import os
import cv2
import torch
import ffmpeg  # type: ignore

from detectron2 import model_zoo  # type: ignore
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.data import MetadataCatalog  # type: ignore
from detectron2.utils.video_visualizer import VideoVisualizer, Visualizer  # type: ignore
from detectron2.utils.logger import setup_logger  # type: ignore


class Lumberjack:  # pylint: disable=too-few-public-methods
    r"""
    Lumberjack automatically extracts tree images from a video or images.

    Args:
        model: Path to the model file.
        output_trees_dir: Directory to save the extracted tree images.
        predict_video_dest_dir: Directory to save the video with predictions in (leave empty to not save).
        visualize: If True, the predictions will be visualized during runtime.
    """

    def __init__(
        self,
        model: str,
        output_trees_dir: str,
        predict_video_dest_dir: Optional[str],
        visualize: bool = True,
    ):
        self.model = model
        self.output_trees_dir = output_trees_dir
        self.predict_video_dest_dir = predict_video_dest_dir
        self.visualize = visualize

        setup_logger()
        torch.cuda.is_available()
        self.logger = setup_logger(name=__name__)

        cfg = get_cfg()
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))

        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 27
        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        cfg.MODEL.MASK_ON = True

        cfg.OUTPUT_DIR = self.output_trees_dir
        cfg.MODEL.WEIGHTS = self.model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        self.predictor_synth = DefaultPredictor(cfg)

        self.tree_metadata = MetadataCatalog.get("my_tree_dataset").set(
            thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"]
        )

    def process_video(  # pylint: disable=too-many-statements, too-many-locals
        self,
        video_path: str,
        corrected: bool = True,
    ):
        r"""
        The process video function takes a video and chops tree images from it.

        Args:
            video: The path to the video file from which to extract tree images.
            corrected: If true, the video can be used as is for extraction. If false, the video will be corrected first.
        """
        if corrected is False:
            directory = os.path.dirname(video_path)
            filename = os.path.basename(video_path)
            new_filename = "corrected_" + filename
            corrected_video_path = os.path.join(directory, new_filename)

            ffmpeg.input(video_path).output(
                corrected_video_path,
                vf="lenscorrection=k1=-0.5:k2=-0.5, " "crop=in_w/3:in_h/3:in_w/3:in_h/3, transpose=1",
            ).run()

        else:
            corrected_video_path = video_path

        vcap = cv2.VideoCapture(corrected_video_path)

        w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(vcap.get(cv2.CAP_PROP_FPS))
        # n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.predict_video_dest_dir is not None:
            dest = os.path.join(
                self.predict_video_dest_dir,
                ("pred_and_track_" + os.path.basename(video_path)),
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video = cv2.VideoWriter(dest, fourcc, 5, (w, h))

        if vcap.isOpened() is False:
            print("Error opening video stream or file")

        vid_vis = VideoVisualizer(metadata=self.tree_metadata)

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
                outputs_pred = self.predictor_synth(crop_frame)

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
                        screenshot_dir = os.path.join(self.output_trees_dir, screenshot_filename)
                        cv2.imwrite(screenshot_dir, cropped_box)  # Save the cropped box
                        i += 1

                out = vid_vis.draw_instance_predictions(crop_frame, outputs_pred["instances"].to("cpu"))

                vid_frame = out.get_image()
                video.write(vid_frame)  # pylint: disable=possibly-used-before-assignment
                if self.visualize:
                    cv2.imshow("frame", vid_frame)
                else:
                    print(f"Frame {nframes}: {len(pred_tree_boxes)} trees detected")

            nframes += 1

        video.release()
        vcap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path: str):  # pylint: disable=too-many-locals
        r"""
        The processing step for a single image.

        Takes an image path and cuts all of the predicted tree bark frames from it and exports it to the output dir.

        Args:
            image_path: The path to the image file from which to extract tree images.
        """

        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
        outputs_pred = self.predictor_synth(image)

        v_synth = Visualizer(
            image[:, :, ::-1],
            metadata=self.tree_metadata,
            scale=1,
        )

        pred_tree_boxes = outputs_pred["instances"].pred_boxes.tensor.cpu().numpy()

        i = 0
        for tree_box in enumerate(pred_tree_boxes):
            box_coords = tree_box[1]  # takes coordinates of the box
            x1, y1, x2, y2 = map(int, box_coords)  # Convert to integer coordinates
            # Crop the frame based on the bounding box coordinates
            cropped_box = image[y1:y2, x1:x2]
            if (x2 - x1) * (y2 - y1) >= 250000:
                angle = (480 - ((x1 + x2) / 2)) / 32  # approximates angle of tree to the camera ortientation
                screenshot_filename = f"box_{i:02d}_angle_{angle:.2f}_{image_name}"
                # Name screenshot by frame index and box index
                screenshot_dir = os.path.join(self.output_trees_dir, screenshot_filename)
                cv2.imwrite(screenshot_dir, cropped_box)  # Save the cropped box
                i += 1

        out_synth = v_synth.draw_instance_predictions(outputs_pred["instances"].to("cpu"))

        if self.visualize:
            cv2.imshow("predictions", out_synth.get_image()[:, :, ::-1])
        else:
            print(f"Image {image_name}: {len(pred_tree_boxes)} trees detected")

    def process_images(self, image_dir: str, cameras: list[int], filetype: str = "jpg"):
        r"""
        Processes all the images from a given directory.

        Args:
            image_dir: The directory containing the images or subdirectories with images to be processed.
            cameras: List of the camera identifiers to be used for filtering images.
        """

        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            if os.path.isfile(image_path):
                for camera in cameras:
                    if re.search(f"_{camera}.{filetype}", image):
                        self.process_image(image_path)
            elif os.path.isdir(image_path):
                self.process_images(image_path, cameras)

        cv2.destroyAllWindows()
