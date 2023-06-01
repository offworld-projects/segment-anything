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
import clip
from sklearn.metrics.pairwise import cosine_similarity

class ObjectDetection:
    """
    The class performs generic multi-object detection on a video stream.
    It uses CLIP pretrained model to make inferences and opencv2 to manage frames.
    """
    def __init__(self, clip_version="ViT-L/14"):
        self.clip_version = clip_version
        self.load_clip_model()

    def load_clip_model(self):
        """
        sets clip models that will be used to get text/image encodings
        
        clip_version (str): string of clip type to load
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(self.clip_version, device=self.device)

    def cosine_dist_img_txt(self, cropped_boxes, class_prompt):
        """
        get list of cosine distances between all cropped boxes single class prompt
        
        cropped_boxes (list): list of boxes that SAM segmented, indices align with returnd list 
        class_prompt (str): class prompt txt
        
        return (list): list of cosine distances of each cropped box to prompt, indices align with cropped_boxes
        """
        cosine_distances = []
        text = clip.tokenize(class_prompt).to(self.device)
        for cropped_box_img in cropped_boxes:  # TODO don't loop through these images
            image = self.clip_preprocess(cropped_box_img).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            img_features = image_features.cpu().detach().numpy()
            txt_features = text_features.cpu().detach().numpy()

            cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
            cosine_distances.append(cosine_dist)
        return cosine_distances

class ImageSegmentation:
    """
    The class performs generic image segentation on a frame.
    It uses segment anything pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot masks overlay with frames.
    """
    def __init__(self, model_type="vit_h"):
        self.object_detection_model = ObjectDetection()

        # move model to GPU for speed if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type

        # three type of model base/large/huge
        if self.model_type == "vit_b":
            self.model_name = "../checkpoints/sam_vit_b_01ec64.pth"
        elif self.model_type == "vit_l":
            self.model_name = "../checkpoints/sam_vit_l_0b3195.pth"
        else:
            self.model_name = "../checkpoints/sam_vit_h_4b8939.pth"

        self.load_model(self.model_name)

        # mask color scheme of n classes in each color
        self.create_color_palette(num_classes=8)

    def load_model(self, sam_checkpoint):
        # load the checkpoint into sam model
        device = self.device
        sam = sam_model_registry[self.model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def segment_image(self, img_np, segmentation_mask):
        """
        returns segmented image 
        
        img_np (numpy array): numpy array of entire image
        segmentation_mask: numpy array of binarys for segmented mask
        
        """
        segmented_img_np = np.zeros_like(img_np)
        segmented_img_np[segmentation_mask] = img_np[segmentation_mask]
        return Image.fromarray(segmented_img_np)

    def get_indices_of_values_above_threshold(self, values, threshold):
        """
        get indicies of bounding box crops that are above cosine distance threshold for prompt
        
        values (list): list of cosine distances betweeen cropped boxes and text
        threshold (float): cosine distance threshold to count as class
        """
        return [i for i, v in enumerate(values) if v > threshold]

    def create_color_palette(self, num_classes):
        # Generate a 3-element color palette tuple
        self.color_palette = np.zeros((num_classes, 3), dtype=np.uint8)
        self.color_palette[:, 0] = np.linspace(0, 255, num_classes, dtype=np.uint8)
        self.color_palette[:, 1] = np.linspace(0, 255, num_classes, dtype=np.uint8)
        self.color_palette[:, 2] = np.linspace(0, 255, num_classes, dtype=np.uint8)
         
    def convert_box_xywh_to_xyxy(self, box):
        """
        convert bounding box from SAM format to x1,y1,x2,y2
        
        box (list) : list in SAM bounding box format of x1,y1,width,hight
        
        return (list): list in bounding box format x1,y1,x2,y2
        """
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def get_bbox_proposal(self, masks):
        """
        get bbox proposal from sam automatic masks predictor, and crop these region out
        """
        cropped_boxes =[]
        for mask in masks:
            cropped_box = self.segment_image(self.cur_img_np, mask["segmentation"]).crop(self.convert_box_xywh_to_xyxy(mask["bbox"]))
            cropped_boxes.append(cropped_box)

        return cropped_boxes

    def classify_masks(self, input_image, masks, classes, obj_threshold=.27):
        """
        masks classification with clip model
        
        masks (list): list of SAM mask dictionaries
        classes (list): list of attempted detected classes
        
        return (numpy array): image numpy array with overlays
        """
        # cut the mask bbox out
        cropped_boxes = self.get_bbox_proposal(masks)
        multi_class_cropped_boxes_score = np.zeros((len(classes), len(cropped_boxes)))

        class_prompts = [f"a photo of a {cls}" for cls in classes]
        img_og = input_image

        print(f"number of crop boxes {len(cropped_boxes)}")
        print(f"number of classes {len(classes)}")

        for idx, class_prompt in enumerate(class_prompts):
            scores = np.array(self.object_detection_model.cosine_dist_img_txt(cropped_boxes, class_prompt))
            filtered_scores = np.zeros_like(scores)
            filtered_scores[scores > obj_threshold] = scores[scores > obj_threshold]   
            multi_class_cropped_boxes_score[idx] = filtered_scores

        img_og = self.image_overlay(classes, img_og, multi_class_cropped_boxes_score, masks)

        return img_og

    def image_overlay(self, classes, img_og, multi_class_cropped_boxes_score, masks):

        FONT_SCALE = 5e-3
        height, width, _ = img_og.shape
        font_scale = min(width, height) * FONT_SCALE
        overlay = copy.deepcopy(img_og)
        alpha = 0.4  # Transparency factor

        # one class for one mask
        indices = np.argmax(multi_class_cropped_boxes_score, axis=0)

        # get the invalid class with 0 value
        invalid_class_idx = np.where(~multi_class_cropped_boxes_score.any(axis=0))[0]
        indices[invalid_class_idx] = -1

        for mask_idx, class_idx in enumerate(indices):
            if class_idx == -1: continue
            cntrs, _ = cv2.findContours(masks[mask_idx]["segmentation"].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            img_og = cv2.polylines(img=img_og, pts=cntrs, isClosed=True, color=(255, 165, 0), thickness=1)
            label = f"{classes[class_idx]}: {round(float(multi_class_cropped_boxes_score[class_idx][mask_idx]), 2)}"
            self.put_label_text(img_og, label, cntrs, width, height, font_scale)
            overlay_mask_color = tuple(int(x) for x in self.color_palette[class_idx])
            cv2.fillPoly(overlay, pts=cntrs, color=overlay_mask_color)

        return cv2.addWeighted(overlay, alpha, img_og, 1 - alpha, 0)

    def put_label_text(self, img_og, label, poly, width, height, font_scale):
        max_poly_idx = 0
        max_poly_area = -1
        for i in range(len(poly)):
            area = cv2.contourArea(poly[i])
            if area > max_poly_area:
               max_poly_idx = i
               max_poly_area = area

        M = cv2.moments(poly[max_poly_idx])
        if M['m00'] == 0:
            center_full_x = 0
            center_full_y = 0
        else:
            center_full_x = M['m10'] / M['m00']
            center_full_y = M['m01'] / M['m00']

        width = poly[i][:,:,1].max() - poly[i][:,:,1].min()  

        font_scale = self.get_optimal_font_scale(label, width)

        cv2.putText(
                    img_og,
                    label,
                    org=(int(center_full_x), int(center_full_y)),
                    fontFace=0,
                    fontScale=font_scale,
                    color=(255, 165, 0),
                    thickness=1
                )

    def get_optimal_font_scale(self, text, width):
        """
        get optimal font scale for writing overlay
        
        text (str): string that will be displayed on overlay 
        width (int): width of box
        
        return (float): optimal font scale size
        """
        for scale in reversed(range(60)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10
        return 1

    def inference(self, frame, stability_score_threshold=.8, predicted_iou_threshold=.8):

        self.cur_img_np = np.asarray(frame)
        masks = self.mask_generator.generate(frame)
        og_masks = masks.copy()

        print(f"number of masks before filter {len(masks)}")
        
        masks = []
        for j in range(0, len(og_masks)):
            if og_masks[j]["stability_score"] > stability_score_threshold and og_masks[j]["predicted_iou"] > predicted_iou_threshold:
                masks.append(og_masks[j])

        print(f"number of masks after filter {len(masks)}")
        return masks
 