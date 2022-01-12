
"""

Auhtor : T. Dhar
Date : 12.01.2021

Objective : To crop and generate images from a High Resolution dataset and downscaling them to generate Low Resolution
            images to train deep learning networks.

"""



import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean




def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right,]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img





def image_downsample(img,scale_fac):                                   #img-----> HxWxC
    x = downscale_local_mean(img[:,:,0], (scale_fac,scale_fac))
    y = downscale_local_mean(img[:,:,1], (scale_fac,scale_fac))
    z = downscale_local_mean(img[:,:,2], (scale_fac,scale_fac))  
    
    return np.stack((x,y,z),axis = 2).astype(img.dtype)
    



def create_dataLR(folder,save_path,size,sc):
    for filenames in os.listdir(folder):
        img_ = io.imread(os.path.join(folder,filenames))
        if img_.shape[0] < size or img_.shape[1] < size:
            continue
        img_ = center_crop(img_,size,size)
        img_ = image_downsample(img_, sc)
        io.imsave(f'{save_path}\\{filenames}',img_)
    print('Images have been saved to {}'.format(save_path))


def create_dataHR(folder,save_path,size):
    for filenames in os.listdir(folder):
        img_ = io.imread(os.path.join(folder,filenames))
        if img_.shape[0] < size or img_.shape[1] < size:
            continue
        img_ = center_crop(img_,size,size)
        io.imsave(f'{save_path}\\{filenames}',img_)
    print('Images have been saved to {}'.format(save_path))



