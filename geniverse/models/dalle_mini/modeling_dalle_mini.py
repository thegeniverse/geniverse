import os
import logging
import math
import random
import numpy as np
from typing import *

import jax
import torch
import torchvision
import PIL
from PIL import Image
from transformers import BartTokenizer  #, CLIPProcessor, FlaxCLIPModel

from geniverse.modeling_utils import ImageGenerator
from forks.dalle_mini import CustomFlaxBartForConditionalGeneration
from forks.vqgan_jax import VQModel

logging.basicConfig(format='%(message)s', level=logging.INFO)

DALLE_REPO = 'flax-community/dalle-mini'
DALLE_COMMIT_ID = '4d34126d0df8bc4a692ae933e3b902a1fa8b6114'

VQGAN_REPO = 'flax-community/vqgan_f16_16384'
VQGAN_COMMIT_ID = '90cc46addd2dd8f5be21586a9a23e1b95aa506a9'


class DalleMini(ImageGenerator):
    """
    Class to optimize images from prompts using the DALL-E Mini 
    model from `https://www.github.com/...`.
    """
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__(device=device, )

        if device is not None:
            self.device = device

        self.bart_tokenizer = BartTokenizer.from_pretrained(
            DALLE_REPO,
            revision=DALLE_COMMIT_ID,
        )
        self.bart_model = CustomFlaxBartForConditionalGeneration.from_pretrained(
            DALLE_REPO, revision=DALLE_COMMIT_ID)

        self.vqgan_model = VQModel.from_pretrained(
            VQGAN_REPO,
            revision=VQGAN_COMMIT_ID,
        )

        # self.clip_model = FlaxCLIPModel.from_pretrained(
        #     "openai/clip-vit-base-patch32")
        # self.clip_processor = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-base-patch32")

    def generate_from_prompt(
        self,
        prompt: str,
        num_res: int = 8,
        **kwargs,
    ) -> Tuple[List[PIL.Image.Image], List[torch.Tensor]]:
        """
        Returns a list of images generated with DALL-E mini.

        Args:
            prompt (str): input prompt.

        Returns:
            List[PIL.Image.Image]: list of optimized images.
        """
        tokenized_prompt = self.bart_tokenizer(
            prompt,
            return_tensors='jax',
            padding='max_length',
            truncation=True,
            max_length=128,
        )

        # create random keys
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        subkeys = jax.random.split(
            key,
            num=num_res,
        )

        print("generate sample predictions")
        z_logits_list = [
            self.bart_model.generate(
                **tokenized_prompt,
                do_sample=True,
                num_beams=1,
                prng_key=subkey,
            ) for subkey in subkeys
        ]

        print("remove first token (BOS)")
        z_logits_list = [img.sequences[..., 1:] for img in z_logits_list]

        decoded_images = [
            self.vqgan_model.decode_code(encoded_image)
            for encoded_image in z_logits_list
        ]

        print("normalize images")
        clipped_images = [img.squeeze().clip(0., 1.) for img in decoded_images]

        print("convert to image")
        gen_img_list = [
            Image.fromarray(np.asarray(img * 255, dtype=np.uint8))
            for img in clipped_images
        ]

        # inputs = self.clip_processor(
        #     text=prompt,
        #     images=gen_img_list,
        #     return_tensors='np',
        # )
        # logits = self.clip_model(**inputs).logits_per_image
        # scores = jax.nn.softmax(
        #     logits,
        #     axis=0,
        # ).squeeze()  # normalize and sum all scores to 1

        # gen_img_list = [gen_img_list[idx] for idx in scores.argsort()[::-1]]

        return gen_img_list, z_logits_list

    # def interpolate(
    #     self,
    #     z_logits_list,
    #     duration_list,
    #     **kwargs,
    # ):
    #     gen_img_list = []
    #     fps = 25

    #     for idx, (z_logits,
    #               duration) in enumerate(zip(z_logits_list, duration_list)):
    #         num_steps = int(duration * fps)
    #         z_logits_1 = z_logits
    #         z_logits_2 = z_logits_list[(idx + 1) % len(z_logits_list)]

    #         for step in range(num_steps):
    #             weight = math.sin(1.5708 * step / num_steps)**2
    #             z_logits = weight * z_logits_2 + (1 - weight) * z_logits_1

    #             z = self.vqgan_model.post_quant_conv(z_logits)
    #             x_rec = self.vqgan_model.decoder(z)
    #             x_rec = (x_rec.clip(-1, 1) + 1) / 2

    #             x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
    #                 x_rec[0])
    #             gen_img_list.append(x_rec_img)

    #             torch.cuda.empty_cache()

    #     return gen_img_list


if __name__ == '__main__':
    prompt = "Landscape of Costa Rica"

    dalle_mini = DalleMini()

    gen_img_list, z_logits_list = dalle_mini.generate_from_prompt(
        prompt=prompt, )

    [img.save(f'{idx}.png') for idx, img in enumerate(gen_img_list)]

    # _gen_img_list, z_logits_list_ = dalle_mini.generate_from_prompt(
    #     prompt="Pokemon of type grass",
    #     lr=lr,
    #     img_save_freq=img_save_freq,
    #     num_generations=num_generations,
    #     num_random_crops=num_random_crops,
    #     init_img_path=init_img_path,
    # )

    # z_logits_interp_list = [z_logits_list[-1], z_logits_list_[-1]]

    # duration_list = [0.7] * len(z_logits_interp_list)
    # interpolate_img_list = dalle_mini.interpolate(
    #     z_logits_list=z_logits_interp_list,
    #     duration_list=duration_list,
    # )
