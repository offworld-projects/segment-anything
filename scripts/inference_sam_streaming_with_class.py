
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
    def __init__(self, clip_version="ViT-B/32"):
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
    The class performs generic image segentation on a video stream.
    It uses segment anything pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot masks overlay with frames.
    """
    def __init__(self, model_name,  streaming_device=0, out_file="output/segmented_video.avi"):
        self.streaming_device = streaming_device
        self.out_file = out_file
        self.object_detection_model = ObjectDetection()

        # move model to GPU for speed if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model(model_name)

        # mask color scheme of 21 classes in each color
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 20 - 1])

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

    def get_overlay_img(self, input_image, masks, classes, obj_threshold=.1):
        """
        get image numpy array overlayed with detections
        
        masks (list): list of SAM mask dictionaries
        classes (list): list of attempted detected classes
        
        return (numpy array): image numpy array with overlays
        """
        # TODO maybe set masks instead of pass in?
        # TODO break this into multiple functions
        FONT_SCALE = 2e-3
        
        # Cut out all masks
        cropped_boxes = []

        for mask in masks:
            cropped_boxes.append(self.segment_image(self.cur_img_np, mask["segmentation"]).crop(self.convert_box_xywh_to_xyxy(mask["bbox"])))

        class_prompts = []
        for cls in classes: 
            class_prompts.append(f"a photo of a {cls}")   
            
        img_og = input_image

        print(f"number of crop boxes {len(cropped_boxes)}")
        print(f"number of classes {len(classes)}")
            
        for idx, class_prompt in enumerate(class_prompts):

            scores = self.object_detection_model.cosine_dist_img_txt(cropped_boxes, class_prompt)
            indices = self.get_indices_of_values_above_threshold(scores, obj_threshold)

            segmentation_masks = []
            polygons = []

            for seg_idx in indices:
                cntrs, _ = cv2.findContours(masks[seg_idx]["segmentation"].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                polygons.append(cntrs)
                segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
                segmentation_masks.append(segmentation_mask_image)

            height, width, _ = img_og.shape

            font_scale = min(width, height) * FONT_SCALE

            for j in range(len(polygons)):
                poly = polygons[j]
                img_og = cv2.polylines(img=img_og, pts=poly, isClosed=True, color=(0, 255, 0), thickness=3)

                label = f"{classes[idx]}__{round(float(scores[indices[j]]), 3)}"

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
                    
                width = poly[i][:,:,1].max() - poly[i][:,:,1].min()  # TODO check if hight or width 
            
                font_scale = self.get_optimal_font_scale(label, width)

                cv2.putText(
                    img_og,
                    label,
                    org=(int(center_full_x), int(center_full_y)),
                    fontFace=0,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=2
                )

        return img_og

    def get_optimal_font_scale(self, text, width):
        """
        get optimal font scale for writing overlay
        
        text (str): string that will be displayed on overlay 
        width (int): width of box
        
        return (float): optimal font scale size
        """
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10
        return 1


    def segment_image(self, img_np, segmentation_mask):
        """
        returns segmented image 
        
        img_np (numpy array): numpy array of entire image
        segmentation_mask: numpy array of binarys for segmented mask
        
        """
        segmented_img_np = np.zeros_like(img_np)
        segmented_img_np[segmentation_mask] = img_np[segmentation_mask]
        segmented_image = Image.fromarray(segmented_img_np)  # pixels black but segment
        return segmented_image

    def get_indices_of_values_above_threshold(self, values, threshold):
        """
        get indicies of bounding box crops that are above cosine distance threshold for prompt
        
        values (list): list of cosine distances betweeen cropped boxes and text
        threshold (float): cosine distance threshold to count as class
        """
        return [i for i, v in enumerate(values) if v > threshold]


    def inference(self, frame, stability_score_threshold=.98, predicted_iou_threshold=.9):

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
            out_frame = self.inference(frame)

            # # convert PIL image back to CV2 image
            # out_frame = (np.float32(out_frame) * 255).round().astype(np.uint8)
            # out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGBA2BGR)

            # overlay the input image with segmentation masks
            classes = ["monitor", "desk", "chair", "keyboard"]
            out_frame = self.get_overlay_img(input_frame, out_frame, classes)
            
            # convert PIL image back to CV2 image
            # out_frame = cv2.cvtColor(np.asarray(out_frame), cv2.COLOR_RGB2BGR)

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

