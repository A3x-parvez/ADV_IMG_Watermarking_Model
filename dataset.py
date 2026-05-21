import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE


class WatermarkDataset(Dataset):

    def __init__(
        self,
        cover_dir,
        watermark_dir,
        augment=False
    ):

        self.cover_dir = cover_dir
        self.watermark_dir = watermark_dir

        self.cover_images = sorted(
            os.listdir(cover_dir)
        )

        self.watermark_images = sorted(
            os.listdir(watermark_dir)
        )

        # =============================================
        # CONSISTENCY CHECK
        # =============================================

        assert len(self.cover_images) == len(
            self.watermark_images
        ), "Dataset size mismatch"

        self.transform = transforms.Compose([

            transforms.Grayscale(),

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor()
        ])

        self.augment = augment

    def __len__(self):

        return len(self.cover_images)

    def __getitem__(self, idx):

        cover_name = self.cover_images[idx]

        watermark_name = self.watermark_images[idx]

        # =============================================
        # FILE MATCH CHECK
        # =============================================

        assert os.path.splitext(cover_name)[0] == \
               os.path.splitext(watermark_name)[0], \
               f"Pair mismatch: {cover_name} != {watermark_name}"

        cover_path = os.path.join(
            self.cover_dir,
            cover_name
        )

        watermark_path = os.path.join(
            self.watermark_dir,
            watermark_name
        )

        cover = Image.open(
            cover_path
        ).convert("L")

        watermark = Image.open(
            watermark_path
        ).convert("L")

        cover = self.transform(cover)

        watermark = self.transform(watermark)

        # =============================================
        # STRICT BINARY WATERMARK
        # =============================================

        watermark = (
            watermark > 0.5
        ).float()

        return (
            cover,
            watermark
        )