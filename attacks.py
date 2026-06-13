import io
import random

import torch
import torchvision.transforms.functional as TF

from torchvision.transforms import GaussianBlur
from PIL import Image


class AttackLayer:

    def __init__(self):

        self.blur = GaussianBlur(
            kernel_size=3
        )

    # =====================================================
    # GAUSSIAN NOISE
    # =====================================================

    def add_gaussian_noise(self, image):

        noise = torch.randn_like(image) * 0.02

        attacked = image + noise

        attacked = torch.clamp(
            attacked,
            0,
            1
        )

        return attacked

    # =====================================================
    # SALT PEPPER NOISE
    # =====================================================

    def add_salt_pepper(self, image):

        attacked = image.clone()

        prob = 0.02

        salt = torch.rand_like(attacked)

        attacked[salt < prob] = 1.0

        attacked[salt > 1 - prob] = 0.0

        return attacked

    # =====================================================
    # BLUR
    # =====================================================

    def add_blur(self, image):

        return self.blur(image)

    # =====================================================
    # ROTATION
    # =====================================================

    def add_rotation(self, image):

        # Batch rotate using torchvision.functional with per-sample angles
        b = image.shape[0]

        angles = [random.uniform(-15, 15) for _ in range(b)]

        rotated = []

        for idx, angle in enumerate(angles):

            single = image[idx]

            rotated.append(TF.rotate(single, angle))

        rotated = torch.stack(rotated, dim=0)

        return rotated

    # =====================================================
    # CROP + RESIZE
    # =====================================================

    def add_crop_resize(self, image):

        b, c, h, w = image.shape

        crop_size = int(h * 0.8)

        top = random.randint(
            0,
            h - crop_size
        )

        left = random.randint(
            0,
            w - crop_size
        )

        cropped = image[
            :,
            :,
            top:top + crop_size,
            left:left + crop_size
        ]

        resized = torch.nn.functional.interpolate(
            cropped,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        return resized

    # =====================================================
    # JPEG COMPRESSION
    # =====================================================

    def add_jpeg_compression(self, image):

        # JPEG via PIL is expensive; approximate compression with differentiable blur + quantization
        blurred = TF.gaussian_blur(image, kernel_size=3)

        # simple 8-bit quantization approximation
        quant = torch.round(blurred * 255.0) / 255.0

        return quant

    # =====================================================
    # RANDOM ATTACK SELECTION
    # =====================================================

    def __call__(self, image):

        p = random.random()

        # =================================================
        # CLEAN IMAGE
        # =================================================

        if p < 0.40:

            return image

        # =================================================
        # RANDOM ATTACK
        # =================================================

        attack = random.choice([

            "jpeg",

            "gaussian",

            "saltpepper",

            "blur",

            "rotate",

            "crop"
        ])

        # =================================================
        # APPLY ATTACK
        # =================================================

        if attack == "jpeg":

            return self.add_jpeg_compression(
                image
            )

        elif attack == "gaussian":

            return self.add_gaussian_noise(
                image
            )

        elif attack == "saltpepper":

            return self.add_salt_pepper(
                image
            )

        elif attack == "blur":

            return self.add_blur(
                image
            )

        elif attack == "rotate":

            return self.add_rotation(
                image
            )

        elif attack == "crop":

            return self.add_crop_resize(
                image
            )

        return image