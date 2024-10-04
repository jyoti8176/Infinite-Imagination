import os
import sys
import numpy as np
import cv2
import torch
import yaml
import glob
import argparse
import tempfile
from PIL import Image
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from omegaconf import OmegaConf
from pathlib import Path
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "lama"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "segment-anything"))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post
from utils.crop_for_replacing import recover_size, resize_and_pad

from segment_anything import SamPredictor, sam_model_registry

class ArgZ:
  def __init__(self, basePath):
    self.sam_ckpt = basePath + "/pretrained_models/sam_vit_h_4b8939.pth"
    self.lama_config = basePath +"/third_party/lama/configs/prediction/default.yaml"
    self.lama_ckpt = basePath + "/pretrained_models/big-lama"

basePath = os.path.dirname(Path(__file__).resolve().parent)
args = ArgZ(basePath)

# build models
model = {}
model_type="vit_h"
ckpt_p=args.sam_ckpt
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt

def build_model():
  global model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # build the sam model
  model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
  model_sam.to(device=device)
  model['sam'] = SamPredictor(model_sam)
  # build the lama model  
  model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

################### STABLE DIFFUSION ####################################
def gen_ai_image(payload):
    # Load base and refiner models with specified parameters
    base_pipe = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base_pipe.to("cuda")
    # refiner_pipe = DiffusionPipeline.from_pretrained(
    #   "stabilityai/stable-diffusion-xl-refiner-1.0",
    #   text_encoder_2=base_pipe.text_encoder_2,
    #   vae=base_pipe.vae,
    #   torch_dtype=torch.float16,
    #   use_safetensors=True,
    #   variant="fp16",
    # )
    # refiner_pipe.to("cuda")
    # image = base_pipe(prompt=payload, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images
    # image = refiner_pipe(prompt=payload, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]
    image = base_pipe(prompt=payload).images[0]
    return image

################### LAMA ####################################
@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location=device)
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location=device)
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    return model

@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

################### STABLE DIFFUSION INPINT ################################
def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        device="cuda"
):
    # Load inpaint model
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16,
    ).to("cuda")
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    img_crop_filled = inpaint_pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled

def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        step: int = 50,
        device="cuda"
):
    # Load inpaint model
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16,
    ).to("cuda")
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = inpaint_pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
    ).images[0]
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized

################### IMAGE SEGMENTATION ####################################
def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        model_type: str,
        ckpt_p: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits

################### GENERAL ####################################
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_points(clicked_points, original_shape, resolution):
    original_height, original_width, _ = original_shape
    original_height = float(original_height)
    original_width = float(original_width)
    
    scale_factor = float(resolution) / min(original_height, original_width)
    resized_points = []
    
    for point in clicked_points:
        x, y, lab = point
        resized_x = int(round(x * scale_factor))
        resized_y = int(round(y * scale_factor))
        resized_point = (resized_x, resized_y, lab)
        resized_points.append(resized_point)
    
    return resized_points

def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features
    orig_h = model['sam'].orig_h
    orig_w = model['sam'].orig_w
    input_h = model['sam'].input_h
    input_w = model['sam'].input_w
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

def get_click_mask(clicked_points, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    # model['sam'].set_image(image)
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    
    # Separate the points and labels
    points, labels = zip(*[(point[:2], point[2])
                            for point in clicked_points])

    # Convert the points and labels to numpy arrays
    input_point = np.array(points)
    input_label = np.array(labels)

    masks, _, _ = model['sam'].predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

def process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, evtindex):
    clicked_coords = evtindex
    x, y = clicked_coords
    label = point_prompt
    lab = 1 if label == "Foreground Point" else 0
    clicked_points.append((x, y, lab))

    input_image = np.array(original_image, dtype=np.uint8)
    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)

    # Update the clicked_points
    resized_points = resize_points(
        clicked_points, input_image.shape, image_resolution
    )
    mask_click_np = get_click_mask(resized_points, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size)

    # Convert mask_click_np to HWC format
    mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0

    mask_image = HWC3(mask_click_np.astype(np.uint8))
    mask_image = cv2.resize(
        mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
    # mask_image = Image.fromarray(mask_image_tmp)

    # Draw circles for all clicked points
    edited_image = input_image
    for x, y, lab in clicked_points:
        # Set the circle color based on the label
        color = (255, 0, 0) if lab == 1 else (0, 0, 255)

        # Draw the circle
        edited_image = cv2.circle(edited_image, (x, y), 10, color, -1)

    # Set the opacity for the mask_image and edited_image
    opacity_mask = 0.5
    opacity_edited = 1.0

    # Combine the edited_image and the mask_image using cv2.addWeighted()
    overlay_image = cv2.addWeighted(
        edited_image,
        opacity_edited,
        (mask_image *
            np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        opacity_mask,
        0,
    )

    return (
        overlay_image,
        # Image.fromarray(overlay_image),
        clicked_points,
        # Image.fromarray(mask_image),
        mask_image
    )

def image_upload(image, image_resolution, clicked_points):
    clicked_points.clear()
    if image is not None:
        np_image = np.array(image, dtype=np.uint8)
        H, W, C = np_image.shape
        np_image = HWC3(np_image)
        np_image = resize_image(np_image, image_resolution)
        features, orig_h, orig_w, input_h, input_w = get_sam_feat(np_image)
        return image, features, orig_h, orig_w, input_h, input_w, clicked_points
    else:
        return None, None, None, None, None, None, clicked_points

def get_inpainted_img(image, mask, image_resolution):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image, mask, lama_config, device=device)
    return img_inpainted

################### ADVANCED IMAGE PROCESSING #################################
def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    point_coords = [w, h]
    point_labels = [1]

    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w

    # model['sam'].set_image(img) # todo : update here for accelerating
    masks, _, _ = model['sam'].predict(
        point_coords=np.array([point_coords]),
        point_labels=np.array(point_labels),
        multimask_output=True,
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

# def get_removed_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
#     lama_config = args.lama_config
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     out = []
#     masks = get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size)
#     for mask in masks:
#         if len(mask.shape)==3:
#             mask = mask[:,:,0]
#         img_inpainted = inpaint_img_with_builded_lama(
#             model['lama'], img, mask, lama_config, device=device)
#         out.append(img_inpainted)
#     return out[2]

def get_removed_img(image, mask, image_resolution):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image, mask, lama_config, device=device)
    return img_inpainted

# def get_replaced_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, text_bg_prompt):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     masks = get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size)
#     mask = masks[2]
#     if len(mask.shape)==3:
#         mask = mask[:,:,0]
#     img_replaced = fill_img_with_sd(
#             img, mask, text_bg_prompt, device=device)
#     img_replaced = img_replaced.astype(np.uint8)
#     return img_replaced

def get_replaced_img(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    # H, W, C = np_image.shape
    # np_image = HWC3(np_image)
    # np_image = resize_image(np_image, image_resolution)

    img_replaced = fill_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced

# def get_replaced_bg_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, text_bg_prompt):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     masks = get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, None)
#     mask = masks[2]
#     if len(mask.shape)==3:
#         mask = mask[:,:,0]
#     img_replaced = replace_img_with_sd(
#             img, mask, text_bg_prompt, device=device)
#     img_replaced = img_replaced.astype(np.uint8)
#     return img_replaced

def get_replaced_bg_img(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    # H, W, C = np_image.shape
    # np_image = HWC3(np_image)
    # np_image = resize_image(np_image, image_resolution)

    img_replaced = replace_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced
  
################### INIT ####################################

build_model()