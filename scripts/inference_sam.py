
import os
import numpy as np

import cv2
import argparse
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from scripts.utils import *

class ImageSegmentationFolder:
    """
    The class performs generic image segentation from a folder.
    It uses segment anything pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot masks overlay with frames.
    """
    def __init__(self, model_type, image_folder_path):
        # initialize segmentation model SAM
        self.sam = ImageSegmentation(model_type=model_type)
        self.image_folder_path = image_folder_path
        self.load_images()

    def get_device(self, device=0):
        # Function creates a streaming object to read the stream frame by frame.
        # return:  OpenCV object to stream video frame by frame.
        cap = cv2.VideoCapture(device)
        assert cap is not None
        return cap

    def load_images(self):
        self.images = []
        for filename in os.listdir(self.image_folder_path):
            img = cv2.imread(os.path.join(self.image_folder_path,filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img)

    def __call__(self):

        classes = ["monitor", "keyboard", "cup", "person", "chair", "backpack", "bicycle"]
        print(f'object classes {classes}')

        output_path = "../out"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for idx, frame in enumerate(self.images):
            input_frame = copy.deepcopy(frame)
            # Convert from OpenCV's BGR format to PyTorch's RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            out_frame = self.sam.inference(frame)

            # overlay the input image with segmentation masks
            out_frame = self.sam.classify_masks(input_frame, out_frame, classes)

            # convert the color channel back to RGB
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(f'../out/image_{idx}.jpg', out_frame)


def main(args):
    print(args)
    ImageSegmentor = ImageSegmentationFolder(args.model_type, args.image_folder)
    ImageSegmentor()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for testing sam')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--image_folder', type=str, default='/home/felix/Pictures/Webcam')
    args = parser.parse_args()
    main(args)

