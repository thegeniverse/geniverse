import abc
import random
import logging
import gc
import math
from typing import *

import torch
import torchvision
import kornia

import PIL
import numpy as np
from PIL import Image
from geniverse_hub import hub_utils


def sinc(x):
    return torch.where(
        x != 0,
        torch.sin(math.pi * x) / (math.pi * x),
        x.new_ones([]),
    )


def lanczos(
    x,
    a,
):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(
    ratio,
    width,
):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio

    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(
    img_batch,
    target_size,
    align_corners=True,
):
    batch_size, num_channels, img_h, img_w = img_batch.shape
    target_img_h, target_img_w = target_size

    img_batch = img_batch.view([batch_size * num_channels, 1, img_h, img_w])

    if target_img_h < img_h:
        kernel_h = lanczos(
            ramp(target_img_h / img_h, 2),
            2,
        ).to(img_batch.device, img_batch.dtype)

        pad_h = (kernel_h.shape[0] - 1) // 2
        img_batch = torch.nn.functional.pad(
            img_batch,
            (0, 0, pad_h, pad_h),
            'reflect',
        )
        img_batch = torch.nn.functional.conv2d(
            img_batch,
            kernel_h[None, None, :, None],
        )

    if target_img_w < img_w:
        kernel_w = lanczos(ramp(target_img_w / img_w, 2), 2).to(
            img_batch.device,
            img_batch.dtype,
        )

        pad_w = (kernel_w.shape[0] - 1) // 2
        img_batch = torch.nn.functional.pad(
            img_batch,
            (pad_w, pad_w, 0, 0),
            'reflect',
        )
        img_batch = torch.nn.functional.conv2d(
            img_batch,
            kernel_w[None, None, None, :],
        )

    img_batch = img_batch.view([batch_size, num_channels, img_h, img_w])

    return torch.nn.functional.interpolate(
        img_batch,
        target_size,
        mode='bicubic',
        align_corners=align_corners,
    )


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        self,
        input_tensor,
        min_value,
        max_value,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.save_for_backward(input_tensor)
        return input_tensor.clamp(min_value, max_value)

    @staticmethod
    def backward(self, grad_in):
        input_tensor, = self.saved_tensors
        return grad_in * (grad_in * (input_tensor - input_tensor.clamp(
            self.min_value, self.max_value)) >= 0, ), None, None


class ImageGenerator(
        torch.nn.Module,
        metaclass=abc.ABCMeta,
):
    """
    This class provides common functionalities among image generators.
    """
    def __init__(
        self,
        device: str = "cuda:0",
        clip_model_name_list: List[str] = [
            "ViT-B/32",
        ],
    ):
        """
        Initializes CLIP, augmentations and set a device.
        """
        super(ImageGenerator, self).__init__()

        self.device = device

        self.clip_model_dict = {}

        self.clip = hub_utils.load_from_hub("clip")

        # jit = True if float(torch.__version__[:3]) < 1.8 else False
        jit = False
        for clip_model_name in clip_model_name_list:
            logging.info(f"LOADING {clip_model_name}...")

            clip_model, clip_preprocess = self.clip.load(
                clip_model_name,
                jit=jit,
                device=self.device,
            )

            clip_model = clip_model.eval()

            clip_preprocess = torchvision.transforms.Compose([
                transform for transform in clip_preprocess.transforms
                if "function" not in repr(transform)
                and "ToTensor" not in repr(transform)
            ])

            self.clip_model_dict[clip_model_name] = {
                "model": clip_model,
                "input_img_size": clip_model.visual.input_resolution,
                "preprocess": clip_preprocess,
            }

        self.active_clip_model_name = random.choice(
            list(self.clip_model_dict.keys()))

        # logging.debug(f"{self.active_clip_model_name} ACTIVE!")

        self.max_clip_img_size = max([
            clip_data["input_img_size"]
            for clip_data in self.clip_model_dict.values()
        ])

        self.supported_loss_types = [
            "cosine_similarity",
            "spherical_distance",
        ]

        self.aug_transform = None

    def add_noise(
        self,
        img_batch: torch.Tensor,
        noise_factor: float = 0.11,
    ) -> torch.Tensor:
        """
        Adds random noise to an image batch.

        Args:
            img_batch (torch.Tensor): image batch where the noise is added.
            noise_factor (float, optional): controls the amount of noise to be added. Defaults to 0.11.

        Returns:
            torch.Tensor: a batch of images with noise.
        """
        noise = noise_factor * torch.rand((img_batch.shape[0], 1, 1, 1)).to(
            self.device) * torch.randn_like(img_batch, requires_grad=False)

        img_batch = img_batch + noise
        # img_batch = ClampWithGrad()(img_batch, 0, 1)
        img_batch = img_batch.clamp(0, 1)

        return img_batch

    def initialize_augmentations(
        self,
        affine_prob: float = 0.8,
        perspective_prob: float = 0.2,
        jitter_prob=0.6,
        grayscale_prob=0.2,
        affine_rotation_degrees: float = 20.,
        affine_translate: Tuple = (0.15, 0.15),
        affine_scale: Tuple = (0.8, 1.2),
        perspective_distorsion: float = 0.15,  #0.05
        jitter_brightness: float = 0.02,
        jitter_contrast: float = 0.02,
        jitter_saturation: float = 0.02,
        jitter_hue: float = 0.01,
    ):
        self.aug_transform = torch.nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.4),
            kornia.augmentation.RandomSharpness(0.3, p=0.1),
            torch.nn.Sequential(
                kornia.augmentation.RandomAffine(
                    degrees=affine_rotation_degrees,
                    translate=affine_translate,
                    scale=affine_scale,
                    p=affine_prob,
                    padding_mode='zeros',
                ),
                kornia.augmentation.RandomPerspective(
                    perspective_distorsion,
                    p=perspective_prob,
                ),
            ),
            kornia.augmentation.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
                p=jitter_prob,
            ),
            kornia.augmentation.RandomGrayscale(p=grayscale_prob),
        ).to(self.device)
        # self.aug_transform = torch.nn.Sequential(
        #     torchvision.transforms.RandomHorizontalFlip(p=0.4, ),
        #     torchvision.transforms.RandomApply(
        #         torch.nn.ModuleList([
        #             torchvision.transforms.RandomAffine(
        #                 degrees=affine_rotation_degrees,
        #                 translate=affine_translate,
        #                 scale=affine_scale,
        #                 # shear=25,
        #                 #
        #             ),
        #         ]),
        #         p=affine_prob,
        #     ),
        #     torchvision.transforms.RandomPerspective(
        #         distortion_scale=perspective_distorsion,
        #         p=perspective_prob,
        #     ),
        #     torchvision.transforms.RandomApply(
        #         torch.nn.ModuleList([
        #             torchvision.transforms.ColorJitter(
        #                 brightness=jitter_brightness,
        #                 contrast=jitter_contrast,
        #                 saturation=jitter_saturation,
        #                 hue=jitter_hue,
        #             ),
        #         ]),
        #         p=jitter_prob,
        #     ),
        # ).to(self.device)

    def augment(
        self,
        img_batch: torch.Tensor,
        target_img_width: int = None,
        target_img_height: int = None,
        num_crops: int = 64,
        noise_factor: float = 0.11,
    ):
        """
        Augments a batch of images using random crops, affine
        transformations and additive noise

        Args:
            img_batch (torch.Tensor): batch of images to augment with shape BxHxWx3.
            target_img_width (int, optional): width of the augmented images. Defaults to img size.
            target_img_height (int, optional): height of the augmented images. Defaults to img size
            num_crops (int, optional): number of augmentations to generate. Defaults to 32.
            noise_factor (float, optional): controls the amount of noise that is added to each crop. Defaults to 0.11.
            pad_downscale (int, optional): represents the fraction of the original image size used to compute the amount of padding to be used. The larger the less padding. Defaults to 3.

        Returns:
            torch.Tensor: augmented batch of images.
        """
        self.active_clip_model_name = random.choice(
            list(self.clip_model_dict.keys()))
        # print(f"{self.active_clip_model_name} ACTIVE! (AUG)")

        if target_img_height is None:
            target_img_height = img_batch.shape[2]
        if target_img_width is None:
            target_img_width = img_batch.shape[3]

        x_pad_percent = 1 / min(1, target_img_width / target_img_height)
        y_pad_percent = 1 / min(1, target_img_height / target_img_width)
        x_pad_size = target_img_width * (x_pad_percent - 1)
        y_pad_size = target_img_height * (y_pad_percent - 1)
        # value = random.choice([0, 256]) / 256
        value = 0
        pad_img_batch = torch.nn.functional.pad(
            img_batch,
            (
                int(x_pad_size / 2),
                int(x_pad_size / 2),
                int(y_pad_size / 2),
                int(y_pad_size / 2),
            ),
            mode='constant',
            value=value,
        )

        max_img_size = max(target_img_width, target_img_height)
        min_img_size = min(target_img_width, target_img_height)

        if self.aug_transform is None:
            self.initialize_augmentations()

        crop_img_list = []
        for crop_idx in range(num_crops):
            crop_size = int(
                torch.normal(
                    .7,
                    .4,
                    (),
                ).clip(min(self.max_clip_img_size / max_img_size, 0.95), 1) *
                max_img_size)

            offsetx = torch.randint(
                0,
                int(target_img_width + x_pad_size - crop_size) + 1,
                (),
            )
            offsety = torch.randint(
                0,
                int(target_img_height + y_pad_size - crop_size) + 1,
                (),
            )

            crop_img = pad_img_batch[:, :, offsety:offsety + crop_size,
                                     offsetx:offsetx + crop_size, ]
            # crop_size = int(
            #     torch.normal(
            #         .7,
            #         .4,
            #         (),
            #     ).clip(min(self.max_clip_img_size / min_img_size, 0.95), 1)
            #     * min_img_size)

            # offsetx = torch.randint(
            #     0,
            #     int(target_img_width - crop_size) + 1,
            #     (),
            # )
            # offsety = torch.randint(
            #     0,
            #     int(target_img_height - crop_size) + 1,
            #     (),
            # )

            # crop_img = img_batch[:, :, offsety:offsety + crop_size,
            #                         offsetx:offsetx + crop_size, ]

            crop_img = torch.nn.functional.interpolate(
                crop_img,
                (self.max_clip_img_size, ) * 2,
                mode='bilinear',
                align_corners=True,
            )

            crop_img_list.append(crop_img)

        img_batch = torch.cat(
            crop_img_list,
            dim=0,
        )
        img_batch = self.aug_transform(img_batch, )

        img_batch = self.add_noise(
            img_batch,
            noise_factor=noise_factor,
        )

        # from PIL import Image
        # import numpy as np

        # for idx in range(img_batch.shape[0]):
        #     Image.fromarray(
        #         np.uint8(
        #             img_batch[idx].permute(1, 2, 0).detach().cpu().numpy() *
        #             255)).save(f'aug_{idx}.jpg')

        return img_batch

    def get_clip_img_encodings(
        self,
        img_batch: torch.Tensor,
        do_normalize: bool = True,
        do_preprocess: bool = True,
    ) -> List[torch.Tensor]:
        """
        Returns the CLIP encoding of an input batch of images.

        Args:
            img_batch (torch.Tensor): input batch of images.
            do_normalize (bool, optional): if `True` the embeddings are normalized. Defaults to True
            do_preprocess (bool, optional): if `True` the input images are normalized and resized. Defaults to True.

        Returns:
            torch.Tensor: batched encodings of the input images.
        """
        img_logits_list = []

        for clip_model_name, clip_model_data in self.clip_model_dict.items():
            if clip_model_name != self.active_clip_model_name:
                # print(f"IMG {self.active_clip_model_name}")
                continue

            if do_preprocess:
                img_batch = clip_model_data["preprocess"](img_batch)
                # img_batch = torch.nn.functional.interpolate(
                #     img_batch,
                #     (clip_model_data["input_img_size"], ) * 2,
                #     mode="bilinear",
                # )

            img_logits = clip_model_data["model"].encode_image(img_batch)

            if do_normalize:
                img_logits = img_logits / img_logits.norm(dim=-1, keepdim=True)

            img_logits_list.append(img_logits, )

        return img_logits_list

    def get_clip_text_encodings(
        self,
        text: Union[str, List[str]],
        do_normalize: bool = True,
    ) -> List[torch.Tensor, ]:
        """
        Returns the CLIP encoding of an input text.

        Args:
            text (str): input text.
            do_normalize (bool, optional): if `True` the embeddings are normalized. Defaults to True

        Returns:
            torch.Tensor: encoding of the input text.
        """
        text_logits_list = []

        if not isinstance(text, list):
            text = [text]

        tokenized_text = self.clip.tokenize(text)
        tokenized_text = tokenized_text.to(self.device).detach().clone()

        for clip_model_name, clip_model_data in self.clip_model_dict.items():
            if clip_model_name != self.active_clip_model_name:
                # print(f"TXT {self.active_clip_model_name}")
                continue

            text_logits = clip_model_data["model"].encode_text(tokenized_text)

            if do_normalize:
                text_logits = text_logits / text_logits.norm(dim=-1,
                                                             keepdim=True)

            text_logits_list.append(text_logits)

        return text_logits_list

    def compute_clip_loss(
        self,
        img_batch: torch.Tensor,
        text: str,
        loss_type: str = 'cosine_similarity',
    ) -> torch.Tensor:
        """
        Computes a distance between the CLIP encodings of a batch
        of images with respect to a text.

        Args:
            img_batch (torch.Tensor): input image batch.
            text (str): input text.
            loss_type (str, optional): Loss type selector. Currently
                loss types: `cosine_similarity` | 
                'spherical_distance'. Defaults to 'cosine_similarity'.

        Returns:
            torch.Tensor: distance value.
        """
        img_logits_list = self.get_clip_img_encodings(img_batch)
        text_logits_list = self.get_clip_text_encodings(text)

        loss = 0
        if loss_type == 'cosine_similarity':
            for img_logits, text_logits in zip(img_logits_list,
                                               text_logits_list):
                loss += -torch.cosine_similarity(
                    text_logits,
                    img_logits,
                ).mean()

        if loss_type == "spherical_distance":
            for img_logits, text_logits in zip(img_logits_list,
                                               text_logits_list):
                loss = (text_logits - img_logits).norm(
                    dim=-1).div(2).arcsin().pow(2).mul(2).mean()

        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def load_img(
        self,
        img_path: str,
    ):
        """
        Load an image from its path and convert it to a tensor.

        Args:
            img_path (str): image path.

        Returns:
            torch.Tensor: tensor representing the input image.
        """
        img_pil = Image.open(img_path)
        img_pil = img_pil.convert('RGB')

        img_tensor = torch.tensor(np.asarray(img_pil)).to(
            self.device).float()[None, :]
        img_tensor /= 255.
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        return img_tensor

    @abc.abstractmethod
    def generate_from_prompt(
        self,
        *args,
        **kwargs,
    ) -> Tuple[List[PIL.Image.Image], List[torch.Tensor]]:
        raise NotImplementedError(
            '`generate_from_prompt` method must be defined by the user.')


class TestImageGenerator(
        ImageGenerator, ):
    def __init__(self, ):
        super().__init__()

    def generate_from_prompt(self, ):
        pass


if __name__ == "__main__":
    image_generator = TestImageGenerator()

    img_batch = torch.randn(1, 3, 200, 400).cuda()
    text = "sup my man"
    loss = image_generator.compute_clip_loss(
        img_batch,
        text,
    )

# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     prompt = "The image of a rainy landscape"

#     anti_prompt = "The image of a sunny landscape"
#     co_prompt = "The image of a cloudy landscape"

#     anti_img_path = "./sunny.jpg"
#     co_img_path = "./rainy.jpg"

#     image_generator = ImageGenerator()

#     prompt_embed = image_generator.get_clip_text_encodings(prompt)
#     anti_prompt_embed = image_generator.get_clip_text_encodings(anti_prompt)
#     co_prompt_embed = image_generator.get_clip_text_encodings(co_prompt)

#     anti_img = image_generator.load_img(anti_img_path)
#     anti_img_embed = image_generator.get_clip_img_encodings(anti_img)

#     co_img = image_generator.load_img(co_img_path)
#     co_img_embed = image_generator.get_clip_img_encodings(co_img)

#     prompt_embed = prompt_embed.clip(-.1, .1)
#     anti_prompt_embed = anti_prompt_embed.clip(-.1, .1)
#     co_prompt_embed = co_prompt_embed.clip(-.1, .1)
#     anti_img_embed = anti_img_embed.clip(-.1, .1)
#     co_img_embed = co_img_embed.clip(-.1, .1)

#     # XXX: SINGLE HIST
#     bins = np.linspace(-.5, .5, 256)

#     plt.figure(figsize=(16, 12))

#     plt.hist(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         bins,
#         alpha=0.5,
#         label='PROMPT',
#     )
#     plt.hist(
#         anti_prompt_embed.detach().cpu().numpy().flatten(),
#         bins,
#         alpha=0.5,
#         label='ANTI PROMPT',
#     )
#     plt.hist(
#         co_prompt_embed.detach().cpu().numpy().flatten(),
#         bins,
#         alpha=0.5,
#         label='CO PROMPT',
#     )
#     plt.hist(
#         anti_img_embed.detach().cpu().numpy().flatten(),
#         bins,
#         alpha=0.5,
#         label='ANTI IMG',
#     )
#     plt.hist(
#         co_img_embed.detach().cpu().numpy().flatten(),
#         bins,
#         alpha=0.5,
#         label='CO IMG',
#     )

#     plt.legend(loc='upper right')

#     plt.savefig('hist.png', dpi=200)

#     # XXX: MULTIPLE HIST
#     bins = np.linspace(-.5, .5, 256)

#     plt.figure(figsize=(16, 12))
#     plt.hist(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#         bins=bins,
#     )
#     plt.hist(
#         anti_prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='ANTI PROMPT',
#         bins=bins,
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('hist-prompt-antiprompt.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.hist(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#         bins=bins,
#     )
#     plt.hist(
#         co_prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='CO PROMPT',
#         bins=bins,
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('hist-prompt-coprompt.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.hist(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#         bins=bins,
#     )
#     plt.hist(
#         anti_img_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='ANTI IMG',
#         bins=bins,
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('hist-prompt-antiimg.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.hist(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#         bins=bins,
#     )
#     plt.hist(
#         co_img_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='CO IMG',
#         bins=bins,
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('hist-prompt-coimg.png', dpi=200)

#     # XXX: MULTIPLE PLOTS

#     plt.figure(figsize=(16, 12))
#     plt.plot(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#     )
#     plt.plot(
#         anti_prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='ANTI PROMPT',
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('plot-prompt-antiprompt.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.plot(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#     )
#     plt.plot(
#         co_prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='CO PROMPT',
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('plot-prompt-coprompt.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.plot(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#     )
#     plt.plot(
#         anti_img_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='ANTI IMG',
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('plot-prompt-antiimg.png', dpi=200)

#     plt.figure(figsize=(16, 12))
#     plt.plot(
#         prompt_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='PROMPT',
#     )
#     plt.plot(
#         co_img_embed.detach().cpu().numpy().flatten(),
#         alpha=0.5,
#         label='CO IMG',
#     )
#     plt.legend(loc='upper right')
#     plt.savefig('plot-prompt-coimg.png', dpi=200)
