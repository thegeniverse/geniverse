import abc
from typing import *

import torch
import torchvision
import clip
import PIL
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


class ImageGenerator(
        torch.nn.Module,
        metaclass=abc.ABCMeta,
):
    """
    This class provides common functionalities among any image 
    generator.
    """
    def __init__(self, ):
        """
        Initializes CLIP, augmentations and set a device.
        """
        super(ImageGenerator, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"USING {self.device}")

        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self.clip_model, _clip_preprocess = clip.load(
            "ViT-B/32",
            jit=jit,
            device=self.device,
        )
        self.clip_model = self.clip_model.eval()

        self.clip_input_img_size = 224

        self.clip_norm_trans = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

        self.aug_transform = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(p=0.7, ),
            torchvision.transforms.RandomApply(
                torch.nn.ModuleList([
                    torchvision.transforms.RandomAffine(
                        degrees=20,
                        translate=(0.1, 0.1),
                        # scale=(0.8, 1),
                        # shear=25,
                        # padding_mode='border',
                    ),
                ]),
                p=0.8,
            ),
            torchvision.transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.4,
            ),
            torchvision.transforms.RandomApply(
                torch.nn.ModuleList([
                    torchvision.transforms.ColorJitter(
                        brightness=0.02,
                        contrast=0.02,
                        saturation=0.02,
                        hue=0.01,
                    ),
                ]),
                p=0.8,
            ),
            # torchvision.transforms.RandomSharpness(
            #     0.3,
            #     p=0.4,
            # ),
        ).to(self.device)

        self.supported_loss_types = [
            "cosine_similarity",
            "spherical_distance",
        ]

    def add_noise(
        self,
        img_batch: torch.Tensor,
        noise_factor: float = 0.11,
    ) -> torch.Tensor:
        # noise = torch.rand((img_batch.shape[0], 1, 1, 1)).to(self.device)
        # noise *= torch.randn_like(img_batch, requires_grad=False)
        noise = noise_factor * torch.rand((img_batch.shape[0], 1, 1, 1)).to(
            self.device) * torch.randn_like(img_batch, requires_grad=False)

        # batch_mask = (torch.rand((img_batch.shape[0])) <= noise_factor).to(self.device)
        # noise *= batch_mask[:, None, None, None]

        img_batch = img_batch + noise
        img_batch = img_batch.clamp(0, 1)

        return img_batch

    def augment(
        self,
        img_batch,
        target_img_width: int = None,
        target_img_height: int = None,
        num_crops=64,
        noise_factor: float = 0.11,
        pad_downscale: int = 3,
    ):
        """
        Augments a batch of images using random crops, affine
        transformations and additive noise

        Args:
            img_batch (torch.Tensor): batch of images to augment with shape BxHxWx3.
            target_img_width (int, optional): width of the augmented images. Defaults to img size.
            target_img_height (int, optional): height of the augmented images. Defaults to img size
            num_crops (int, optional): Number of augmentations to generate. Defaults to 32.

        Returns:
            torch.Tensor: augmented batch of images.
        """
        if target_img_height is None:
            target_img_height = img_batch.shape[2]
        if target_img_width is None:
            target_img_width = img_batch.shape[3]

        x_pad_size = target_img_width // pad_downscale
        y_pad_size = target_img_height // pad_downscale
        img_batch = torch.nn.functional.pad(
            img_batch,
            (
                x_pad_size,
                x_pad_size,
                y_pad_size,
                y_pad_size,
            ),
            mode='constant',
            value=0,
        )

        min_img_size = min(target_img_width, target_img_height)

        aug_img_batch = self.aug_transform(img_batch)

        augmented_img_list = []
        for crop_idx in range(num_crops):

            crop_size = int(
                torch.normal(
                    1,
                    .2,
                    (),
                ).clip(.4, 1.2) * min_img_size)

            # if crop < num_crops - 4:
            #     if crop > num_crops - 8:
            #         crop_size = int(min_img_size * 1.2)

            offsetx = torch.randint(
                0,
                int(target_img_width + target_img_width * 2 / pad_downscale -
                    crop_size),
                (),
            )
            offsety = torch.randint(
                0,
                int(target_img_height + target_img_height * 2 / pad_downscale -
                    crop_size),
                (),
            )
            augmented_img = aug_img_batch[:, :, offsety:offsety + crop_size,
                                          offsetx:offsetx + crop_size, ]

            # else:
            #     augmented_img = aug_img_batch[:, :, x_pad_size:-x_pad_size,
            #                                   y_pad_size:-y_pad_size]

            augmented_img = torch.nn.functional.interpolate(
                augmented_img,
                (self.clip_input_img_size, ) * 2,
                mode='bilinear',
                align_corners=True,
            )

            augmented_img_list.append(augmented_img)

        img_batch = torch.cat(augmented_img_list, 0)

        # img_list = []
        # for _ in range(num_crops):
        #     # img_batch = img_batch.repeat(num_crops, 1, 1, 1)
        #     img_list.append(self.aug_transform(img_batch))
        # img_batch = torch.cat(img_list)

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
    ) -> torch.Tensor:
        """
        Returns the CLIP encoding of an input batch of images.

        Args:
            img_batch (torch.Tensor): input batch of images.
            do_normalize (bool, optional): if `True` the embeddings are normalized. Defaults to True
            do_preprocess (bool, optional): if `True` the input images are normalized and resized. Defaults to True.

        Returns:
            torch.Tensor: batched encodings of the input images.
        """
        if do_preprocess:
            img_batch = self.clip_norm_trans(img_batch)
            img_batch = torch.nn.functional.upsample_bilinear(
                img_batch,
                (self.clip_input_img_size, self.clip_input_img_size),
            )

        img_logits = self.clip_model.encode_image(img_batch)

        if do_normalize:
            img_logits = img_logits / img_logits.norm(dim=-1, keepdim=True)

        return img_logits

    def get_clip_text_encodings(
        self,
        text: str,
        do_normalize: bool = True,
    ):
        """
        Returns the CLIP encoding of an input text.

        Args:
            text (str): input text.
            do_normalize (bool, optional): if `True` the embeddings are normalized. Defaults to True

        Returns:
            torch.Tensor: encoding of the input text.
        """
        tokenized_text = clip.tokenize([text])
        tokenized_text = tokenized_text.to(self.device).detach().clone()

        text_logits = self.clip_model.encode_text(tokenized_text)

        if do_normalize:
            text_logits = text_logits / text_logits.norm(dim=-1, keepdim=True)

        return text_logits

    def compute_clip_loss(
        self,
        img_batch: torch.Tensor,
        text: str,
        loss_type: str = 'cosine_similarity',
        loss_clip_value: float = None,
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
            loss_clip_value (float, optional): value for thresholding
                the CLIP embeddings. Defaults to None.

        Returns:
            torch.Tensor: distance value.
        """
        img_logits = self.get_clip_img_encodings(img_batch)
        text_logits = self.get_clip_text_encodings(text)

        if loss_clip_value is not None:
            img_logits = img_logits.clip(-loss_clip_value, loss_clip_value)
            text_logits = text_logits.clip(-loss_clip_value, loss_clip_value)

        loss = 0
        if loss_type == 'cosine_similarity':
            loss += -torch.cosine_similarity(text_logits, img_logits).mean()
        if loss_type == "spherical_distance":
            loss = (text_logits - img_logits).norm(
                dim=-1).div(2).arcsin().pow(2).mul(2).mean()

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


if __name__ == "__main__":
    prompt = "The image of a rainy landscape"

    anti_prompt = "The image of a sunny landscape"
    co_prompt = "The image of a cloudy landscape"

    anti_img_path = "./sunny.jpg"
    co_img_path = "./rainy.jpg"

    image_generator = ImageGenerator()

    prompt_embed = image_generator.get_clip_text_encodings(prompt)
    anti_prompt_embed = image_generator.get_clip_text_encodings(anti_prompt)
    co_prompt_embed = image_generator.get_clip_text_encodings(co_prompt)

    anti_img = image_generator.load_img(anti_img_path)
    anti_img_embed = image_generator.get_clip_img_encodings(anti_img)

    co_img = image_generator.load_img(co_img_path)
    co_img_embed = image_generator.get_clip_img_encodings(co_img)

    prompt_embed = prompt_embed.clip(-.1, .1)
    anti_prompt_embed = anti_prompt_embed.clip(-.1, .1)
    co_prompt_embed = co_prompt_embed.clip(-.1, .1)
    anti_img_embed = anti_img_embed.clip(-.1, .1)
    co_img_embed = co_img_embed.clip(-.1, .1)

    # XXX: SINGLE HIST
    bins = np.linspace(-.5, .5, 256)

    plt.figure(figsize=(16, 12))

    plt.hist(
        prompt_embed.detach().cpu().numpy().flatten(),
        bins,
        alpha=0.5,
        label='PROMPT',
    )
    plt.hist(
        anti_prompt_embed.detach().cpu().numpy().flatten(),
        bins,
        alpha=0.5,
        label='ANTI PROMPT',
    )
    plt.hist(
        co_prompt_embed.detach().cpu().numpy().flatten(),
        bins,
        alpha=0.5,
        label='CO PROMPT',
    )
    plt.hist(
        anti_img_embed.detach().cpu().numpy().flatten(),
        bins,
        alpha=0.5,
        label='ANTI IMG',
    )
    plt.hist(
        co_img_embed.detach().cpu().numpy().flatten(),
        bins,
        alpha=0.5,
        label='CO IMG',
    )

    plt.legend(loc='upper right')

    plt.savefig('hist.png', dpi=200)

    # XXX: MULTIPLE HIST
    bins = np.linspace(-.5, .5, 256)

    plt.figure(figsize=(16, 12))
    plt.hist(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
        bins=bins,
    )
    plt.hist(
        anti_prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='ANTI PROMPT',
        bins=bins,
    )
    plt.legend(loc='upper right')
    plt.savefig('hist-prompt-antiprompt.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.hist(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
        bins=bins,
    )
    plt.hist(
        co_prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='CO PROMPT',
        bins=bins,
    )
    plt.legend(loc='upper right')
    plt.savefig('hist-prompt-coprompt.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.hist(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
        bins=bins,
    )
    plt.hist(
        anti_img_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='ANTI IMG',
        bins=bins,
    )
    plt.legend(loc='upper right')
    plt.savefig('hist-prompt-antiimg.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.hist(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
        bins=bins,
    )
    plt.hist(
        co_img_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='CO IMG',
        bins=bins,
    )
    plt.legend(loc='upper right')
    plt.savefig('hist-prompt-coimg.png', dpi=200)

    # XXX: MULTIPLE PLOTS

    plt.figure(figsize=(16, 12))
    plt.plot(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
    )
    plt.plot(
        anti_prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='ANTI PROMPT',
    )
    plt.legend(loc='upper right')
    plt.savefig('plot-prompt-antiprompt.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.plot(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
    )
    plt.plot(
        co_prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='CO PROMPT',
    )
    plt.legend(loc='upper right')
    plt.savefig('plot-prompt-coprompt.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.plot(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
    )
    plt.plot(
        anti_img_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='ANTI IMG',
    )
    plt.legend(loc='upper right')
    plt.savefig('plot-prompt-antiimg.png', dpi=200)

    plt.figure(figsize=(16, 12))
    plt.plot(
        prompt_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='PROMPT',
    )
    plt.plot(
        co_img_embed.detach().cpu().numpy().flatten(),
        alpha=0.5,
        label='CO IMG',
    )
    plt.legend(loc='upper right')
    plt.savefig('plot-prompt-coimg.png', dpi=200)
