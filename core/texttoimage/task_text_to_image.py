import numpy as np
import torch
from diffusers import DiffusionPipeline

from ..base import Task

class TaskTextToImage(Task):

    def __init__(self):
        super().__init__()
        # Initialize Stable Diffusion
        self.base_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.base_pipe.to("cuda")

    def executeTask(self, payload):
        image = self.base_pipe(prompt=payload).images[0]
        np_image = np.array(image, dtype=np.uint8)
        return np_image
