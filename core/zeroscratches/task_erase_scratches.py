from .zeroscratches import EraseScratches

from ..utils import array2image
from ..base import Task


class TaskEraseScratches(Task):

    def __init__(self):
        super().__init__()
        self.scratchEraser = EraseScratches()

    def executeTask(self, image):
        image = array2image(image)
        return self.scratchEraser.erase(image)

