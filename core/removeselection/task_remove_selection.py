from ..coreutils import get_removed_img
from ..base import Task

class TaskRemoveSelection(Task):

    def __init__(self):
        super().__init__()

    def executeTask(self, image, mask, image_resolution):
        new_img = get_removed_img(image, mask, image_resolution)
        return new_img