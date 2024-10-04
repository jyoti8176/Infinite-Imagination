import numpy as np

from .siggraph17 import siggraph17
from .util import postprocess_tens, preprocess_img

from ..utils import array2image
from ..base import Task

class TaskImageColorizer(Task):
    def __init__(self):
        super().__init__()

    def executeTask(self, image):
        #image = array2image(image)
        img = siggraph17(pretrained=True).eval()
        oimg = image #np.asarray(image)
        if(oimg.ndim == 2):
            oimg = np.tile(oimg[:,:,None], 3)
        (tens_l_orig, tens_l_rs) = preprocess_img(oimg)

        output_img = postprocess_tens(tens_l_orig, img(tens_l_rs).cpu())
        return np.array(output_img)