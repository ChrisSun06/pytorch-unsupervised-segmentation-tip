#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from scipy.ndimage import convolve

use_cuda = torch.cuda.is_available()

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
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1.5, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# load image
im = cv2.imread(args.input)
data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) ) 
# image size: 640 * 480
# print("reshaped image data shape", data.shape)
# print("reshaped image data", data)
if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.'+args.input.split('.')[-1],'_scribble.png'),-1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
    inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
    inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
    target_scr = torch.from_numpy( mask.astype(np.int) )
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable( target_scr )
    # set minLabels
    # args.minLabels = len(mask_inds)

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    outputHP = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy,HPy_target)
    lhpz = loss_hpz(HPz,HPz_target)

    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        # cv2.imshow( "output", im_target_rgb )
        # cv2.waitKey(10)

    # loss 
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + args.stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + args.stepsize_con * (lhpy + lhpz)
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# THRESH = 12
GREEN_LOWER = [0, 150, 0] # [45, 50, 20]
GREEN_UPPER = [150, 255, 150]
GREEN_BASE = [0, 150, 0]
NORM = 300

def find_plant(im, im_target):
    image_data = np.array([im])[0]
    reshaped_image_data = image_data.reshape((im_target.shape[0], 3))
    reshaped_im_target = im_target.reshape((im.shape[0], im.shape[1]))
    mapped = {} # map from label to color
    percent_mapped = {} # map from label to green percentage
    labels = np.unique(im_target)
    for x in labels:
        mask = np.where(reshaped_im_target==x)
        # number of pixels in the label
        # count = len(mask[0])
        # if count < 0.05 * im_target.shape[0]:
        #     mapped[x] = [255,255,255]
        #     continue
        masked_image = image_data[mask[0], mask[1]]
        green_percent = process_green(masked_image, len(mask[0]))
        if green_percent < 3:
            mapped[x] = [255,255,255]
        else:
            mapped[x] = None
            percent_mapped[x] = green_percent
        # if green_percent > THRESH:
        #     mapped[x] = None   # np.array([173,255,47])
        # else:
        #     mapped[x] = [255,255,255]
    # remove the classes with the lowest green percentage
    # sorted_percent_mapped = sorted(percent_mapped.items(), key=lambda kv: kv[1])
    # mapped[sorted_percent_mapped[0][0]] = [255,255,255]
    # print(sorted_percent_mapped)

    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([mapped[c] if mapped[c] is not None else np.array(reshaped_image_data[i]) for i,c in enumerate(im_target)])
    process_image(im_target_rgb.reshape( im.shape ), im)
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( "output_plant.png", im_target_rgb )

def process_image(image, im):
    kernel = np.ones((10, 10))
    binary_image = np.all(image == [255,255,255], axis=-1).astype(int)
    convolved_image = convolve(binary_image, kernel)
    white_pixels = np.argwhere(np.all(image == [255,255,255], axis=-1))
    print("image", image.shape)
    print("im", np.array([im])[0].shape)
    for pixel in white_pixels:
        row, col = pixel
        if convolved_image[row, col] < np.sum(kernel):
            image[row, col] = np.array([im])[0][row][col]

    return image

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
    # norm_mask = np.linalg.norm(img - np.array(GREEN_BASE), axis=1) <= NORM
    # norm_pixel_count = np.count_nonzero(norm_mask)
    # return int(norm_pixel_count / count * 100)
    

# find plant vs. non-plant
find_plant(im, im_target)

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
cv2.imwrite( "output.png", im_target_rgb )
