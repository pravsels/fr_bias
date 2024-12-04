# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import torch.utils.data as data
import os
from PIL import Image
from utils import preprocess_image


class CelebASegmentation(data.Dataset):
  CLASSES = ['background' ,'skin','nose',
             'eye_g','l_eye','r_eye','l_brow','r_brow',
             'l_ear','r_ear',
             'mouth','u_lip','l_lip',
             'hair','hat',
             'ear_r','neck_l','neck',
             'cloth']

  def __init__(self, root, transform=None, crop_size=None):
    self.root = root
    self.transform = transform
    self.crop_size = crop_size

    self.images = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".jpg", ".png", ".jpeg"))]


  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _img=_img.resize((513,513),Image.BILINEAR)
    _img = preprocess_image(_img,flip=False,scale=None,crop=(self.crop_size, self.crop_size))

    if self.transform is not None:
        _img = self.transform(_img)

    return _img

  def __len__(self):
    return len(self.images)
  
  
class FlexibleImageSegmentation(data.Dataset):
  CLASSES = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
               'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

  def __init__(
      self,
      root: str,
      output_dir: str = None, 
      transform=None,
      crop_size=None,
      is_valid_file=None
  ):
      self.root = root
      self.output_dir = output_dir
      self.transform = transform
      self.crop_size = crop_size
      self.is_valid_file = is_valid_file or (lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg')))
      
      # Get all images and their status
      self.images, self.stats = self._make_dataset(self.root, self.is_valid_file)
        
      if len(self.images) == 0 and self.stats['total'] == 0:
          raise RuntimeError(f"Found 0 files in subfolders of: {self.root}\n"
                             "Supported extensions are: .jpg, .jpeg, .png")
        
      # Print detailed statistics
      print("\nDataset Statistics:")
      print(f"Total images found: {self.stats['total']}")
      if self.stats['processed'] > 0:
          print(f"Already processed: {self.stats['processed']} ({self.stats['processed']/self.stats['total']*100:.1f}%)")
      if self.stats['corrupted'] > 0:
          print(f"Corrupted images: {self.stats['corrupted']} ({self.stats['corrupted']/self.stats['total']*100:.1f}%)")
      print(f"Remaining to process: {len(self.images)} ({len(self.images)/self.stats['total']*100:.1f}%)")
      print()
    
  def _make_dataset(self, dir_path, is_valid_file):
      images = []
      stats = {'total': 0, 'processed': 0, 'corrupted': 0}
      dir_path = os.path.expanduser(dir_path)
        
      for root, _, fnames in sorted(os.walk(dir_path, followlinks=True)):
          for fname in sorted(fnames):
              path = os.path.join(root, fname)
              if is_valid_file(path):
                  stats['total'] += 1
                    
                  # Check if mask exists
                  if self.output_dir:
                      relative_path = os.path.relpath(path, self.root)
                      mask_path = os.path.join(
                          self.output_dir,
                          os.path.splitext(relative_path)[0] + '_mask.png'
                      )
                      if os.path.exists(mask_path):
                          stats['processed'] += 1
                          continue
                    
                  # Try to open the image to check if it's corrupted
                  try:
                      with Image.open(path) as img:
                          img.verify()  # Verify it's actually an image
                      images.append(path)
                      if len(images) > 200:
                        return images, stats  
                  except (IOError, OSError, Image.UnidentifiedImageError) as e:
                      stats['corrupted'] += 1
                      print(f"Skipping corrupted image {path}: {str(e)}")
        
      return images, stats

  def __getitem__(self, index):
      img_path = self.images[index]
      _img = Image.open(img_path).convert('RGB')
      _img = _img.resize((513, 513), Image.BILINEAR)
      _img = preprocess_image(_img, flip=False, scale=None, crop=(self.crop_size, self.crop_size))
      
      if self.transform is not None:
          _img = self.transform(_img)
      
      return _img

  def __len__(self):
      return len(self.images)

