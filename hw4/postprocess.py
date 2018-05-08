import os
import numpy as np
from PIL import Image

fileID = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5]
No = [5, 7, 8, 12, 3, 12, 15, 19, 21, 0, 1, 4, 5, 6, 15, 19, 13, 14, 15, 16, 19, 27, 29, 0, 2, 5, 6, 20, 21, 31, 6, 14]

def main():
    width, height = 64, 64
    n_col, n_row = 8, 4
    

    whole = Image.new('RGB', (n_col*width, n_row*height))

    for i in range(len(fileID)):
        im = np.asarray(Image.open('sample/{:03d}.jpg'.format(fileID[i])))
        offset_h = (No[i] // 8) * height
        offset_w = (No[i] % 8) * width

        im = Image.fromarray(im[offset_h:offset_h+height, offset_w:offset_w+width, :])

        whole.paste(im, ((i%8)*width, (i//8)*height, (i%8+1)*width, (i//8+1)*height))
    whole.save('gan_sample.jpg')


if __name__ == '__main__':
   main()