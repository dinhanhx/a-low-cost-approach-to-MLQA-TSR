from typing import NamedTuple

import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

MAX_IMAGE_HEIGHT = 1536
MAX_IMAGE_WIDTH = 1536


class Resolution(NamedTuple):
    height: int
    width: int


class ImageFeatureExtraction:
    def __init__(self, model_name: str = "nvidia/C-RADIOv2-B") -> None:
        self.model_name = model_name
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self.image_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            self.image_model.eval().cuda()

    def get_target_size_for_large_image(self, source_resolution: Resolution) -> Resolution:
        if source_resolution.width >= source_resolution.height:
            new_width = MAX_IMAGE_WIDTH
            new_height = new_width * source_resolution.height // source_resolution.width
        if source_resolution.height >= source_resolution.width:
            new_height = MAX_IMAGE_HEIGHT
            new_width = new_height * source_resolution.width // source_resolution.height
        return self.image_model.get_nearest_supported_resolution(height=new_height, width=new_width)

    def get_target_size_for_image(self, source_resolution: Resolution) -> Resolution:
        if source_resolution.width > MAX_IMAGE_WIDTH or source_resolution.height > MAX_IMAGE_HEIGHT:
            return self.get_target_size_for_large_image(source_resolution)

        return self.image_model.get_nearest_supported_resolution(
            height=source_resolution.height,
            width=source_resolution.width,
        )

    def infer_single(self, image: Image.Image):
        target_size = self.get_target_size_for_image(Resolution(image.height, image.width))

        model_inputs = self.image_processor(
            images=[image],
            return_tensors="pt",
            do_resize=True,
            size={"height": target_size.height, "width": target_size.width},
        )
        model_inputs = model_inputs.to("cuda")

        with torch.no_grad():
            outputs = self.image_model(model_inputs.pixel_values)

        # RADIO will return a tuple with two tensors.
        # The summary is similar to the cls_token in ViT and is meant to represent the general concept.
        # It has shape (B,C) with B being the batch dimension, and C being some number of channels.
        # The spatial_features represent more localized content.
        general_feature = outputs.summary[0].cpu().numpy().tolist()

        torch.cuda.empty_cache()
        return general_feature
