# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ImageMaskDatasetGenerator.py
# 2024/05/01 antillia.com

import os
import sys
import shutil
import cv2
import glob
import traceback

from scipy.ndimage.filters import gaussian_filter
import numpy as np

class ImageMaskDatasetGenerator:
  
  def __init__(self, width=512, height=512, 
                input_images_dir= "./2d_images",
                input_masks_dir = "./2d_masks" ,
                output_dir      = "./Lung_master/",
                augmentation    = True):
    self.W          = width
    self.H          = height
    self.input_images_dir  = input_images_dir
    self.input_masks_dir  = input_masks_dir

    self.output_dir = output_dir
    self.augmentation= augmentation
    if self.augmentation:
      self.hflip    = True
      self.vflip    = False
      self.rotation = True
      self.ANGLES   = [5, 360]
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.01, 0.02]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)

      self.resize = True
      self.resize_ratio = 0.8

    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    self.output_images_dir = os.path.join(self.output_dir, "images")
    if not os.path.exists(self.output_images_dir):
      os.makedirs(self.output_images_dir)

    self.output_masks_dir  = os.path.join(self.output_dir, "masks")
    if not os.path.exists(self.output_masks_dir):
      os.makedirs(self.output_masks_dir)

  def generate(self):
    mask_files = glob.glob(self.input_masks_dir + "/*.tif")
    for mask_file in mask_files:
      mask = cv2.imread(mask_file)
      mask = cv2.resize(mask, (self.W, self.H))
      basename = os.path.basename(mask_file)
      #basename = basename.replace(".tif", ".jpg")
      output_mask_file = os.path.join(self.output_masks_dir, basename)
      cv2.imwrite(output_mask_file, mask)
      print("=== Saved{}".format(output_mask_file))
      self.augment(mask, basename, self.output_masks_dir, border=(0, 0, 0), mask=True)

    image_files = glob.glob(self.input_images_dir + "/*.tif")
    for image_file in image_files:
      image  = cv2.imread(image_file)
      image = cv2.resize(image, (self.W, self.H))
      basename = os.path.basename(image_file)
      #basename = basename.replace(".png", ".jpg")
      output_image_file = os.path.join(self.output_images_dir, basename)
      cv2.imwrite(output_image_file, image)
      print("=== Saved{}".format(output_image_file))
      self.augment(image, basename, self.output_images_dir, border=(255,255,255), mask=False)

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir, mask)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))
      
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir, mask):
    #print("----shrink shape {}".format(image.shape))
    h, w, c = image.shape
    rh = int(h* self.resize_ratio)
    rw = int(w * self.resize_ratio)
    resized = cv2.resize(image, (rw, rh))
    h1, w1  = resized.shape[:2]
    ph = int((h - h1)/2)
    pw = int((w - w1)/2)
    # black background
    background = np.zeros((h, w, c), np.uint8)
    if mask == False:
      # white background
      background = np.ones((h, w, c), np.uint8) * 255
    # paste resized to background
    background[ph:ph+h1, pw:pw+w1] = resized
    filename = "shrinked_" + str(self.resize_ratio) + "_" + basename
    output_file = os.path.join(output_dir, filename)    

    cv2.imwrite(output_file, background)
    print("=== Saved shrinked image file{}".format(output_file))


if __name__ == "__main__":
  
  try:
    # generate 
    #   ./Lung-master
    #      +-- images 
    #      +-- masks 
    #  from the orginal 2d_images and 2d_masks dataset with augmentation=True
    #
    generator = ImageMaskDatasetGenerator(width=512, height=512, 
                                          input_images_dir= "./2d_images", 
                                          input_masks_dir = "./2d_masks",
                                          output_dir      = "./Lung-master/", 
                                          augmentation    = True)
    generator.generate()
    
  except:
    traceback.print_exc()
