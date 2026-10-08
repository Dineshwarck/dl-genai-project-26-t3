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