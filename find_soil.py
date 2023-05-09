#from __future__ import print_function
import argparse
import torch
import cv2
import numpy as np
import torch.nn.init
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1.5, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
parser.add_argument('--image_src', metavar="IMG_SRC", default="sh_1k_data/images/val", type=str, 
                        help="image folder for images to be segmented")
args = parser.parse_args()


YELLOW_LOWER = [150, 100, 0]
YELLOW_UPPER = [255, 255, 150]

GREEN_LOWER = [0, 150, 0] # [45, 50, 20]
GREEN_UPPER = [150, 255, 150]

def find_soil(image_path):
    # load image
    im = cv2.imread(image_path)
    image_data = np.array([im])[0]
    w, h = np.shape(image_data)[:-1]
    reshaped_image_data = image_data.reshape((w * h, 3))    # 1d image data
    # yellow_percent = process_yellow(reshaped_image_data, len(reshaped_image_data))
    green_percent = process_green(reshaped_image_data, len(reshaped_image_data))
    print(green_percent)
    if green_percent < 1:
        return True
    else:
        return False


def process_green(img, count):
    green_mask = np.logical_and.reduce((
        img[:,0] >= GREEN_LOWER[0],
        img[:,1] >= GREEN_LOWER[1],
        img[:,2] >= GREEN_LOWER[2],
        img[:,0] <= GREEN_UPPER[0],
        img[:,1] <= GREEN_UPPER[1],
        img[:,2] <= GREEN_UPPER[2]
))
    green_pixel_count = np.count_nonzero(green_mask)
    return int(green_pixel_count / count * 100)

def process_yellow(img, count):
    yellow_mask = np.logical_and.reduce((
        img[:,0] >= YELLOW_LOWER[0],
        img[:,1] >= YELLOW_LOWER[1],
        img[:,2] >= YELLOW_LOWER[2],
        img[:,0] <= YELLOW_UPPER[0],
        img[:,1] <= YELLOW_UPPER[1],
        img[:,2] <= YELLOW_UPPER[2]
))
    
    yellow_pixel_count = np.count_nonzero(yellow_mask)
    return int(yellow_pixel_count / count * 100)
    




if __name__ == "__main__":
    for img in tqdm(os.listdir(args.image_src)):
      image_path = os.path.join(args.image_src,img)
      if find_soil(image_path):
          # save output image
          with open("soil_images.txt", "a") as file:
              file.write(image_path + "\n")