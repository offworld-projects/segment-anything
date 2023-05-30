
import os
import sys
import torch
import numpy as np
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import cv2
import argparse
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def load_images(image_folder_path):
    images = []
    for filename in os.listdir(image_folder_path):
        img = cv2.imread(os.path.join(image_folder_path,filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

def load_model(sam_checkpoint):

    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator

def inference(mask_generator, image_list):

    for image in image_list:
        masks = mask_generator.generate(image)
        print(len(masks))
        print(masks[0].keys())
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()

def main(args):
    print(args)

    mask_generator = load_model(sam_checkpoint=args.model_name)
    images = load_images(image_folder_path=args.image_folder)
    inference(mask_generator, images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for testing deeplabv3')
    parser.add_argument('--model_name', type=str, default='../checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--image_folder', type=str, default='/home/felix/repos/low_light_image_enhancer/img')
    args = parser.parse_args()
    main(args)

