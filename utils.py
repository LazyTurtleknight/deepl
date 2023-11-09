import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

def plot(sample, data_dir):

  plt.figure(figsize=(5,4))
  ax = plt.subplot(2,2,1)
  plt.imshow(np.asarray(Image.open(os.path.join(data_dir, sample['sat_image_path'].iloc[0]))))
  plt.gray()
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)

  ax = plt.subplot(2,2,2)
  plt.imshow(np.asarray(Image.open(os.path.join(data_dir, sample['mask_path'].iloc[0]))))
  plt.gray()
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)

  plt.show()