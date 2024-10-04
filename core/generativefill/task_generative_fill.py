from ..coreutils import get_replaced_img
from ..base import Task

class TaskGenerativeFill(Task):

    def __init__(self):
        super().__init__()        

    def executeTask(self, origin_image, click_mask, image_resolution, adv_text_prompt):
        new_img = get_replaced_img(origin_image, click_mask, image_resolution, adv_text_prompt)
        return new_img
