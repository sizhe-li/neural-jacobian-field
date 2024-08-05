import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange
from jaxtyping import Bool, Float
from pycocotools.coco import COCO
from torch import Tensor
from torchvision.transforms import functional as F


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        images: List[Float[Tensor, "... H W"]],
        pix_map: Float[Tensor, "... 2 H W"],
    ):
        H, W = images[0].shape[-2:]

        if torch.rand(1) < self.p:
            images = [F.hflip(x) for x in images]

            pix_map[..., 0, :, :] = (W - 1) - pix_map[..., 0, :, :]

        return images, pix_map


class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        images: List[Float[Tensor, "... H W"]],
        pix_map: Float[Tensor, "... 2 H W"],
    ):
        H, W = images[0].shape[-2:]

        if torch.rand(1) < self.p:
            images = [F.vflip(x) for x in images]

            pix_map[..., 1, :, :] = (H - 1) - pix_map[..., 1, :, :]

        return images, pix_map


class ZeroMaskPatchedImage(nn.Module):
    def __init__(self, patch_size: int = 20, mask_ratio: float = 0.5):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, image: Float[Tensor, "c h w"]):
        # convert to patch
        image = image.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )

        l, k = image.shape[1], image.shape[2]
        L = k * l

        image = rearrange(image, "c l k h w -> (l k) c h w")

        # randomly select mask_ratio length out of L
        rand_idx = torch.randperm(L)[: int(L * self.mask_ratio)]
        image[rand_idx] = torch.zeros_like(image[rand_idx])

        # convert back to image
        image = rearrange(image, "(l k) c h w -> c l h k w", l=l, k=k)
        image = rearrange(image, "c l h k w -> c (l h) (k w)")

        return image


class RandomBackground(nn.Module):
    def __init__(
        self,
        p=0.5,
        # test_mode: bool = False,
        # area_to_occlude: float = 0.5,
        # location_to_occlude: float = 0.5,
    ):
        super().__init__()
        self.p = p

        # coco related
        self.root_dir = Path("/weka/scratch/user/sizhe086/project/data/coco2017")
        self.overlay_threshold = 0.5
        self.min_area_threshold = 10000
        self.max_area_threshold = 50000

        annotation_filename = self.root_dir / "annotations/instances_train2017.json"
        self.coco = COCO(annotation_filename)
        self.coco_img_keys = list(self.coco.imgs.keys())
        self.coco_cat_ids = self.coco.getCatIds()

    def randomize_bkgd(
        self,
        img: Float[Tensor, "... 3 H W"],
        mask: Bool[Tensor, "... H W"],
    ):
        rgb_mask = torch.zeros_like(img)
        rgb_mask[..., 0, :, :] = rgb_mask[..., 1, :, :] = rgb_mask[..., 2, :, :] = mask

        img = img * rgb_mask

        # Next, domain randomize all non-masked parts of image
        rgb_mask_comp = torch.ones_like(img) - rgb_mask

        random_rgb_img = get_random_image(img.shape[-3:])
        random_rgb_img = (random_rgb_img / 255.0).float()

        random_rgb_bkgd = random_rgb_img * rgb_mask_comp
        img = img + random_rgb_bkgd

        return img

    def randomize_coco(
        self,
        input_image: Float[Tensor, "... 3 H W"],
        mask: Bool[Tensor, "... H W"],
    ):
        mask_area_before = torch.sum(mask, dim=(-2, -1))
        mask_area_thresh = mask_area_before * self.overlay_threshold

        # get random image
        random_coco_image_index = random.choice(self.coco_img_keys)
        coco_image_info = self.coco.imgs[random_coco_image_index]
        coco_image_filename = coco_image_info["file_name"]

        coco_image = Image.open(self.root_dir / "train2017" / coco_image_filename)

        annotations_ids = self.coco.getAnnIds(
            imgIds=coco_image_info["id"], catIds=self.coco_cat_ids, iscrowd=None
        )
        annotations = self.coco.loadAnns(annotations_ids)

        if len(annotations) == 0:
            return input_image

        # choose random annotation
        random_ann_index = torch.randint(0, len(annotations), (1,)).item()
        annotation = annotations[random_ann_index]

        # load bbox
        if annotation["area"] < self.min_area_threshold:
            return input_image

        x_min, y_min, width, height = annotation["bbox"]
        x_max, y_max = x_min + width, y_min + height
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # crop image
        cropped_coco_image = coco_image.crop((x_min, y_min, x_max, y_max))
        if annotation["area"] > self.max_area_threshold:
            # find a size that will satisfy the max area threshold
            # choose a random threshold inbetween min and max
            rand_threshold = torch.randint(
                self.min_area_threshold, self.max_area_threshold, (1,)
            ).item()
            new_width = int(width * (rand_threshold / annotation["area"]))
            new_height = int(height * (rand_threshold / annotation["area"]))
            cropped_coco_image = cropped_coco_image.resize(
                (new_width, new_height), Image.BILINEAR
            )

        # cropped_coco_image needs to be smaller than input_image
        if (
            cropped_coco_image.size[0] >= input_image.shape[-1]
            or cropped_coco_image.size[1] >= input_image.shape[-2]
        ):
            # make it smaller by a factor of 2
            cropped_coco_image = cropped_coco_image.resize(
                (
                    cropped_coco_image.size[0] // 2,
                    cropped_coco_image.size[1] // 2,
                ),
                Image.BILINEAR,
            )

        cropped_coco_image = F.to_tensor(cropped_coco_image)

        # randomly overlay coco_image on top of img

        # get random location
        H, W = input_image.shape[-2:]
        h, w = cropped_coco_image.shape[-2:]
        x = torch.randint(0, H - h, (1,)).item()
        y = torch.randint(0, W - w, (1,)).item()

        # get new mask
        mask[..., x : x + h, y : y + w] = 0

        # get new mask area
        mask_area_after = torch.sum(mask, dim=(-2, -1))

        # if mask area is too small, then revert back to original image
        if torch.any(mask_area_after < mask_area_thresh):
            return input_image

        # overlay
        input_image[..., x : x + h, y : y + w] = cropped_coco_image

        return input_image

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            img = self.randomize_bkgd(img, mask)

        if torch.rand(1) < self.p:
            img = self.randomize_coco(img, mask)

        return img


def get_random_image(shape):
    if torch.rand(1) < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape)
        rgb2 = get_random_solid_color_image(shape)
        vertical = (torch.rand(1) < 0.5).item()
        rand_image = get_gradient_image(rgb1, rgb2, vertical)

    if torch.rand(1) < 0.5:
        return rand_image
    else:
        return add_noise(rand_image)


def get_random_solid_color_image(shape):
    rand_rgb = (torch.rand(3) * 255).to(torch.uint8)
    return torch.ones(shape, dtype=torch.uint8) * rand_rgb[:, None, None]


def get_gradient_image(rgb1, rgb2, vertical):
    """
    Interpolates between two images rgb1 and rgb2

    :param rgb1, rgb2: two numpy arrays of shape (3,H,W)

    :return interpolated image:
    :rtype: same as rgb1 and rgb2
    """
    bitmap = torch.zeros_like(rgb1)
    H, W = rgb1.shape[-2:]
    if vertical:
        alpha = torch.tile(torch.linspace(0, 1, H)[:, None], (1, W))
    else:
        alpha = torch.tile(torch.linspace(0, 1, W), (H, 1))

    alpha = alpha[None, ...].repeat(3, 1, 1)
    bitmap = rgb2 * alpha + rgb1 * (1.0 - alpha)

    return bitmap


def add_noise(rgb_image):
    """
    Adds noise, and subtracts noise to the rgb_image

    :param rgb_image: image to which noise will be added
    :type rgb_image: numpy array of shape (H,W,3)

    :return image with noise:
    :rtype: same as rgb_image

    ## Note: do not need to clamp, since uint8 will just overflow -- not bad
    """
    max_noise_to_add_or_subtract = 50
    return (
        rgb_image
        + get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)
        - get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)
    )


def get_random_entire_image(shape, max_pixel_uint8):
    """
    Expects something like shape=(480,640,3)

    Returns an array of that shape, with values in range [0..max_pixel_uint8)

    :param max_pixel_uint8: maximum value in the image
    :type max_pixel_uint8: int

    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return (torch.rand(size=shape) * max_pixel_uint8).to(torch.uint8)
