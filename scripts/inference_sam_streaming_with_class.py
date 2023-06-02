
import time
import copy
import numpy as np
import cv2
from .utils import *

class ImageSegmentationStreaming:
    """
    The class performs generic image segentation on a video stream.
    It uses segment anything pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot masks overlay with frames.
    """
    def __init__(self, model_type, streaming_device=0, out_file="output/segmented_video.avi"):
        # initialize segmentation model SAM
        self.sam = ImageSegmentation(model_type=model_type)
        self.streaming_device = streaming_device
        self.out_file = out_file

    def get_device(self, device=0):
        # Function creates a streaming object to read the stream frame by frame.
        # return:  OpenCV object to stream video frame by frame.
        cap = cv2.VideoCapture(device)
        assert cap is not None
        return cap

    def __call__(self):
        # specify the object class to segment
        classes = ["monitor", "keyboard", "cup", "person"]
        print(f'object classes {classes}')

        # create streaming service for application
        player = self.get_device(self.streaming_device) 
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
            out_frame = self.sam.inference(frame)

            # overlay the input image with segmentation masks
            out_frame = self.sam.classify_masks(input_frame, out_frame, classes)

            end_time = time.time()
            fps += 1/np.round(end_time - start_time, 3)
            time.sleep(1.0)

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
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--streaming_device', type=int, default=0)
    args = parser.parse_args()
    main(args)

