# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# This code is a modification of the main.py file
# from the https://github.com/chenxi116/DeepLabv3.pytorch repository

import argparse
import os
import requests
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

import deeplab
from data_loader import CelebASegmentation
from utils import download_file
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type=int, default=256,
					help='segmentation output size')
parser.add_argument('--workers', type=int, default=4,
					help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=2,
					help='batch size of images processed')
args = parser.parse_args()


resnet_file_spec = dict(
                        file_path='deeplab_model/R-101-GN-WS.pth.tar', 
                        )
deeplab_file_spec = dict(
                         file_path='deeplab_model/deeplab_model.pth', 
                         )

def main():
    resolution = args.resolution
    batch_size = args.batch_size

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    model_fname = 'deeplab_model/deeplab_model.pth'
    dataset_root = '../../datasets/synthpar'

    assert os.path.isdir(dataset_root)

    dataset = CelebASegmentation(dataset_root, 
                                 crop_size=513)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=args.workers)
    
    if not os.path.isfile(resnet_file_spec['file_path']):
        print('Downloading backbone Resnet Model parameters')
        with requests.Session() as session:
            download_file(session, resnet_file_spec)

        print('Done!')

    model = getattr(deeplab, 'resnet101')(
                    pretrained=True,
                    num_classes=len(dataset.CLASSES),
                    num_groups=32,
                    weight_std=True,
                    beta=False)

    model = model.cuda()
    model.eval()
    if not os.path.isfile(deeplab_file_spec['file_path']):
        print('Downloading DeeplabV3 Model parameters')
        with requests.Session() as session:
            download_file(session, deeplab_file_spec)

        print('Done!')

    checkpoint = torch.load(model_fname)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)

    output_dir = os.path.join(dataset_root, 'masks')
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, image_batch in enumerate(tqdm(dataloader, desc='skinning faces...')):
        inputs = image_batch.cuda()
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

        for i in range(preds.shape[0]):
            img_index = batch_idx * batch_size + i 
            if img_index >= len(dataset):
                break

            img_name = os.path.basename(dataset.images[img_index])
            mask_pred = Image.fromarray(preds[i])
            mask_pred = mask_pred.resize((resolution, resolution), Image.NEAREST)

            output_path = os.path.join(output_dir, img_name.split('.')[0] + '_mask.png')
            mask_pred.save(output_path)

if __name__ == "__main__":
  main()

