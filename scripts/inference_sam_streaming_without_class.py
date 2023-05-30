
import os
import sys
import time
import copy
import torch
import numpy as np
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import cv2
import argparse
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


class ImageSegmentation:
    """
    The class performs generic image segentation on a video stream.
    It uses segment anything pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot masks overlay with frames.
    """
    def __init__(self, model_name, streaming_device=0, out_file="output/segmented_video.avi"):
        self.streaming_device = streaming_device
        self.out_file = out_file

        # move model to GPU for speed if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model(model_name)


    def get_device(self, device=0):
        # Function creates a streaming object to read the stream frame by frame.
        # return:  OpenCV object to stream video frame by frame.
        cap = cv2.VideoCapture(device)
        assert cap is not None
        return cap

    def load_model(self, sam_checkpoint):
        # load the checkpoint into sam model
        model_type = "vit_b"
        device = self.device

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
       

    def image_overlay(self, image, segmented_image):
        alpha = 1  # transparency for the original image
        beta  = 0.8  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
    
        image = np.array(image)
        cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
        return image

    def show_anns(self, anns):
        # generate the RGBA masks with 4 channels
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img

    def inference(self, frame):
        masks = self.mask_generator.generate(frame)
        out_frame  = self.show_anns(masks)
        return out_frame 
       
    def __call__(self):
        player = self.get_device(self.streaming_device) # create streaming service for application
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")

        # start saving segmentation result video
        # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        fc = 0
        fps = 0
        tfc = int(player.get(cv2.CAP_PROP_FRAME_COUNT))
        tfcc = 0
        while True:
            fc += 1
            start_time = time.time()
            ret, frame = player.read()
            input_frame = copy.deepcopy(frame)
            if not ret:
                break

            # Convert from OpenCV's BGR format to PyTorch's RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            out_frame = self.inference(frame) # 4-channel RGBA

            # convert PIL image back to CV2 image
            out_frame = (np.float32(out_frame) * 255).round().astype(np.uint8)
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGBA2BGR)

            # overlay the input image with segmentation masks
            # import pdb; pdb.set_trace()
            out_frame = self.image_overlay(input_frame, out_frame)

            # convert PIL image back to CV2 image
            out_frame = cv2.cvtColor(np.asarray(out_frame), cv2.COLOR_RGB2BGR)

            end_time = time.time()
            fps += 1/np.round(end_time - start_time, 3)
            if fc == 10:
                fps = int(fps / 10)
                tfcc += fc
                fc = 0
                per_com = int(tfcc / tfc * 100)
                print(f"Frames Per Second : {fps}")
            # out.write(out_frame)
            cv2.imshow('segmented frame', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        player.release()
        cv2.destroyAllWindows() 


def main(args):
    print(args)
    ImageSegmentor = ImageSegmentation(args.model_name, args.streaming_device)
    ImageSegmentor()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for testing segment anything model')
    parser.add_argument('--model_name', type=str, default='../checkpoints/sam_vit_b_01ec64.pth')
    parser.add_argument('--streaming_device', type=int, default=0)
    args = parser.parse_args()
    main(args)

