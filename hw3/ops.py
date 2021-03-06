from __future__ import print_function, division

import random
import numpy as np
from utils import get_file_paths, get_image, vgg_sub_mean, mask_preprocess, mask_postprocess

def batch_gen(dir, batch_size):
    content_path, mask_path = get_file_paths(dir)    
    content_path.sort()
    mask_path.sort()

    while True:
        index = random.sample(range(1, len(content_path)), batch_size)
        try:
            contents = [vgg_sub_mean(get_image(content_path[i])) for i in index]
            masks = [mask_preprocess(get_image(mask_path[i])) for i in index]

            contents = np.asarray(contents, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)

        except Exception as err:
            print("\nError: {}".format(err))
            continue

        yield contents, masks
