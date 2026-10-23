import numpy as np
import pandas as pd


def rle_to_mask(rle_string,height,width):
    if pd.isna(rle_string) or rle_string == "":
        return np.zeros((height,width),dtype=np.uint8)
    rle_int=[int(x) for x in rle_string.split()] 
    mask_1d = np.zeros((height*width),dtype=np.uint8)
    for i in range(0,len(rle_int),2):
        start_pixel = rle_int[i] - 1
        length = rle_int[i+1]
        #print(start_pixel,length)
        mask_1d[start_pixel:start_pixel+length] = 1
    mask_2d = mask_1d.reshape((height,width),order='F')
    return mask_2d

def mask_to_rle(mask):
  """mask: 2D numpy array, 1 - defect, 0 - background

  Returns run-length encoding string (1-indexed, Fortran/column-major order).
  """
  pixels = mask.flatten(order='F')
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)
