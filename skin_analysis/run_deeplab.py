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
from data_loader import FlexibleImageSegmentation
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
                        # file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', 
                        file_path='deeplab_model/R-101-GN-WS.pth.tar', 
                        # file_size=178260167, 
                        # file_md5='aa48cc3d3ba3b7ac357c1489b169eb32'
                        )

deeplab_file_spec = dict(
                        # file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', 
                         file_path='deeplab_model/deeplab_model.pth', 
                        #  file_size=464446305, 
                        #  file_md5='8e8345b1b9d95e02780f9bed76cc0293'
                         )


def main():
    resolution = args.resolution
    batch_size = args.batch_size

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    model_fname = 'deeplab_model/deeplab_model.pth'
    dataset_root = '../../test_dataset'

    assert os.path.isdir(dataset_root)

    dataset = FlexibleImageSegmentation(dataset_root, 
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

            img_path = dataset.images[img_index]
            relative_path = os.path.relpath(img_path, dataset_root)
            mask_filename = os.path.splitext(os.path.basename(img_path))[0] + '_mask.png'
            
            # Construct the output path while preserving the folder structure
            output_path = os.path.join(output_dir, os.path.dirname(relative_path), mask_filename)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            mask_pred = Image.fromarray(preds[i])
            mask_pred = mask_pred.resize((resolution, resolution), Image.NEAREST)
            mask_pred.save(output_path)

if __name__ == "__main__":
  main()

