import pc_util as pc_util;
import argparse
import math
import h5py
import numpy as np
import os
import sys
from PIL import Image
# from show3d_balls import showpoints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

path = "./output/"
for i in range(195, 226):
    p = path + str(i) + ".out"
    savep = "./images/" + str(i) + ".png"
    pc = np.loadtxt(p).astype(np.float32)
    im_array = pc_util.point_cloud_three_views(pc)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save(savep)