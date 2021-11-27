import os
import yaml
import math
import logging
import requests
from typing import *

import torch
import torchvision
import numpy as np
import omegaconf
import PIL
from PIL import Image
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from forks.taming_transformers.taming.models.vqgan import VQModel, GumbelVQ
from geniverse.modeling_utils import ImageGenerator
from scipy.stats import norm

logging.basicConfig(format='%(message)s', level=logging.INFO)

VQGAN_CKPT_DICT = {
    "imagenet-16384":
    r"https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    "openimages-8192":
    r"https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
}
VQGAN_CONFIG_DICT = {
    "imagenet-16384":
    r"https://raw.githubusercontent.com/vipermu/taming-transformers/master/configs/imagenet-16384.yaml",
    "openimages-8192":
    r"https://raw.githubusercontent.com/vipermu/taming-transformers/master/configs/openimages-8192.yaml",
}

FPS = 25


class TamingDecoder(ImageGenerator):
    """
    Image generator that leverages the decoder of VQGAN to 
    optimize images. This code uses the original implementation
    from https://github.com/CompVis/taming-transformers.
    """
    def __init__(
        self,
        device: str = 'cuda',
        model_name: str = "imagenet-16384",
        **kwargs,
    ) -> None:
        """
        Downloads the VQGAN model and loads a pre-defined 
        configuration.

        Args:
            device (str, optional): defaults to 'cuda'.
        """
        super().__init__(
            device=device,
            **kwargs,
        )

        if device is not None:
            self.device = device

        modeling_dir = os.path.dirname(os.path.abspath(__file__))
        modeling_cache_dir = os.path.join(modeling_dir, ".modeling_cache")
        os.makedirs(modeling_cache_dir, exist_ok=True)

        modeling_ckpt_path = os.path.join(modeling_cache_dir,
                                          f'{model_name}.ckpt')
        if not os.path.exists(modeling_ckpt_path):
            modeling_ckpt_url = VQGAN_CKPT_DICT[model_name]

            logging.info(
                f"Downloading pre-trained weights for VQ-GAN from {modeling_ckpt_url}"
            )
            results = requests.get(modeling_ckpt_url, allow_redirects=True)

            with open(modeling_ckpt_path, "wb") as ckpt_file:
                ckpt_file.write(results.content)

        # TODO: update the url with our own config using the correct paths
        modeling_config_path = os.path.join(modeling_cache_dir,
                                            f'{model_name}.yaml')
        if not os.path.exists(modeling_config_path):
            modeling_config_url = VQGAN_CONFIG_DICT[model_name]

            logging.info(
                f"Downloading `{model_name}.yaml` from vipermu taming-transformers fork"
            )
            results = requests.get(modeling_config_url, allow_redirects=True)

            with open(modeling_config_path, "wb") as yaml_file:
                yaml_file.write(results.content)

        vqgan_config_xl = self.load_config(
            config_path=modeling_config_path,
            display=False,
        )
        self.vqgan_model = self.load_vqgan(
            vqgan_config_xl,
            ckpt_path=modeling_ckpt_path,
        ).to(self.device)

    @staticmethod
    def load_config(
        config_path: str,
        display=False,
    ) -> omegaconf.dictconfig.DictConfig:
        """
        Loads a VQGAN configuration file from a path or URL.

        Args:
            config_path (str): local path or URL of the config file.
            display (bool, optional): if `True` the configuration is 
                printed. Defaults to False.

        Returns:
            omegaconf.dictconfig.DictConfig: configuration dictionary.
        """
        config = OmegaConf.load(config_path)

        if display:
            logging.info(yaml.dump(OmegaConf.to_container(config)))

        return config

    @staticmethod
    def load_vqgan(
        config: omegaconf.dictconfig.DictConfig,
        ckpt_path: str = None,
    ) -> VQModel:
        """
        Load a VQGAN model from a config file and a ckpt path 
        where a VQGAN model is saved.

        Args:
            config ([type]): VQGAN model config.
            ckpt_path ([type], optional): path of a saved model. 
                Defaults to None.

        Returns:
            VQModel: 
                loaded VQGAN model.
        """
        if "GumbelVQ" in config.model.target:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)

        if ckpt_path is not None:
            # XXX: check wtf is going on here
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)

        return model.eval()

    def get_img_from_latents(
        self,
        z_logits: torch.Tensor,
        target_img_height: int = None,
        target_img_width: int = None,
    ) -> torch.Tensor:
        z = self.vqgan_model.post_quant_conv(z_logits)
        img_rec = self.vqgan_model.decoder(z)
        img_rec = (img_rec.clip(-1, 1) + 1) / 2

        if target_img_height is not None and target_img_width is not None:
            img_rec = torch.nn.functional.interpolate(
                img_rec,
                (target_img_height, target_img_width),
                mode="bilinear",
            )

        return img_rec

    @staticmethod
    def get_random_latents(
        target_img_height,
        target_img_width,
        batch_size=1,
    ):
        embed_height = target_img_height // 16
        embed_width = target_img_width // 16

        z_logits = .5 * torch.randn(
            batch_size,
            256,
            embed_height,
            embed_width,
        )

        z_logits = torch.sinh(1.9 * torch.arcsinh(z_logits))

        return z_logits

    def get_latents_from_img(
        self,
        img: Union[torch.Tensor, PIL.Image.Image],
        num_rec_steps: int = 0,
    ):
        if torch.is_tensor(img):
            img_tensor = img
            img_tensor = img_tensor * 2 - 1
        else:
            img = img.convert('RGB')
            img_tensor = torch.tensor(np.asarray(img))
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = (img_tensor / 255.) * 2 - 1

        if len(img_tensor.shape) <= 3:
            img_tensor = img_tensor[None, ::]

        img_tensor = img_tensor.float().to(self.device)

        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            (int((img_tensor.shape[2] / 16) * 16),
             int((img_tensor.shape[3] / 16) * 16)),
            mode="bilinear",
        )

        z, _, [_, _, _indices] = self.vqgan_model.encode(img_tensor)

        z = z.to(self.device)

        rec_optimizer = None
        for rec_idx in range(num_rec_steps):
            if rec_idx == 0:
                if not torch.is_tensor(img):
                    img = img.convert('RGB')
                    img = torch.tensor(np.asarray(img))
                    img = img_tensor.permute(2, 0, 1)
                    img = (img / 255.)

                img = img.detach()
                z = torch.nn.Parameter(z).detach()
                z.requires_grad = True
                rec_optimizer = torch.optim.AdamW(
                    params=[z],
                    lr=0.01,
                    betas=(0.9, 0.999),
                    weight_decay=0.1,
                )

                rec_loss_weight = img_tensor.shape[2] * img_tensor.shape[3]

            img_rec = self.get_img_from_latents(z, )
            rec_loss = rec_loss_weight * torch.nn.functional.mse_loss(
                img_rec,
                img,
            )

            logging.info(f"Rec loss {rec_loss}")

            rec_optimizer.zero_grad()
            rec_loss.backward()
            rec_optimizer.step()

        return z

    def generate_from_prompt(
        self,
        prompt: str,
        lr: float = 0.5,
        target_img_height: int = 256,
        target_img_width: int = 256,
        num_steps: int = 200,
        num_augmentations: int = 32,
        init_img_path: str = None,
        loss_type: str = 'cosine_similarity',
        generation_cb=None,
        init_embed: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[List[PIL.Image.Image], List[torch.Tensor]]:
        """
        Returns a list of images generated while optimizing the
        input of a pre-trained VQGAN decoder with a prompt.

        Args:
            prompt (str): input prompt.
            lr (float, optional): learning rate. Defaults to 0.5.
            target_img_size (int, optional): image size of the 
                generated images. Defaults to 256.
            num_steps (int, optional): number of optimization 
                steps. Defaults to 200.
            num_augmentations (int, optional): number of augmented 
                images to use. Defaults to 20.
            init_img_path (str, optional): path for an image to use
                as the starting point of the optimization. Defaults 
                to None.
            loss_type (str, optional): either 'cosine_similarity' 
                or 'spherical_distance'. Defaults to 
                'cosine_similarity'.
            generation_cb (function, optional): if provided, the 
                function is executed for every optimization step. The
                inputs of this function will be:  loss, step, rec_img, z_logits and step
            init_embed (torh.Tensor, optional): initial embedding 
                From where to start the generation. Defaults to None.

        Returns:
            Tuple[List[PIL.Image.Image], List[torch.Tensor]]: list 
                of optimized images and their respective logits.
        """
        # TODO: implement batching
        batch_size = 1

        if loss_type not in self.supported_loss_types:
            print(
                (f"ERROR! Loss type {loss_type} not recognized. "
                 f"Only {' or '.join(self.supported_loss_types)} supported."))

            return

        if init_embed is not None:
            z_logits = init_embed.to(self.device)

        elif init_img_path is not None:
            init_img = Image.open(init_img_path, )
            init_img = init_img.resize((target_img_height, target_img_width))
            z = self.get_latents_from_img(init_img)

            z_logits = z.to(self.device)

            logging.warn(
                f"TARGET RES CHANGED TO {target_img_height}x{target_img_width}"
            )

        else:
            z_logits = self.get_random_latents(
                target_img_height=target_img_height,
                target_img_width=target_img_width,
                batch_size=batch_size,
            ).to(self.device)

        z_logits = torch.nn.Parameter(z_logits)

        optimizer = torch.optim.AdamW(
            params=[z_logits],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        gen_img_list = []
        z_logits_list = []
        for step in range(num_steps):
            loss = 0

            z_logits_list.append(z_logits.detach().clone())

            rec_img = self.get_img_from_latents(
                z_logits,
                target_img_height=target_img_height,
                target_img_width=target_img_width,
            )

            x_rec_stacked = self.augment(
                rec_img,
                num_crops=num_augmentations,
                target_img_width=target_img_width,
                target_img_height=target_img_height,
            )

            clip_loss = 10 * self.compute_clip_loss(
                x_rec_stacked,
                prompt,
                loss_type,
            )

            loss += clip_loss

            # if init_img is not None:
            #     loss += -10 * torch.cosine_similarity(z_logits,
            #                                           img_z_logits).mean()
            # if init_img is not None:
            #     loss += -10 * torch.cosine_similarity(
            #         self.get_clip_img_encodings(rec_img), clip_img_z_logits).mean()

            logging.info(f"\nIteration {step} of {num_steps}")
            logging.info(f"Loss {round(float(loss.data), 2)}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                rec_img[0])
            gen_img_list.append(x_rec_img)

            if generation_cb is not None:
                generation_cb(
                    loss=loss,
                    step=step,
                    rec_img=rec_img,
                    latents=z_logits,
                )

            torch.cuda.empty_cache()

        return gen_img_list, z_logits_list

    def interpolate(
        self,
        latents_list: List[torch.Tensor],
        duration_list: List[float],
        interpolation_type: str = "sinusoidal",  #sinusoidal | linear
        loop: bool = True,
        **kwargs,
    ) -> List[PIL.Image.Image]:
        """
        Returns a list of PIL images corresponding to the 
        generations produced by interpolating the values
        from `latents_list`.

        Args:
            latents_list (List[torch.Tensor]): list of VQVAE 
                intermediate embeddings.
            duration_list (List[float]): list with the duration of
                the interpolation between each consecutive tensor. 
                The last value represent the duration between the 
                last tensor and the initial.

        Returns:
            List[PIL.Image.Image]: list of the resulting generated images.
        """
        z_logits_list = latents_list

        gen_img_list = []

        for idx, (z_logits,
                  duration) in enumerate(zip(z_logits_list, duration_list)):

            if idx == len(z_logits_list) - 1 and not loop:
                break

            num_steps = int(duration * FPS)
            z_logits_1 = z_logits
            z_logits_2 = z_logits_list[(idx + 1) % len(z_logits_list)]

            for step in range(num_steps):
                if step == num_steps - 1 and num_steps > 1:
                    weight = 1

                else:
                    if interpolation_type == "linear":
                        if num_steps - 1 > 0:
                            weight = step / (num_steps - 1)
                        else:
                            weight = 0

                    else:
                        weight = math.sin(1.5708 * step / num_steps)**2

                z_logits = weight * z_logits_2 + (1 - weight) * z_logits_1

                rec_img = self.get_img_from_latents(z_logits, )

                x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    rec_img[0])
                gen_img_list.append(x_rec_img)

                torch.cuda.empty_cache()

        return gen_img_list

    def zoom(
        self,
        prompt: str,
        init_latents: torch.Tensor,
        lr: float = 0.05,
        num_generations: int = 64,
        num_zoom_interp_steps=1,
        num_zoom_train_steps=1,
        zoom_offset=4,
    ):
        gen_img_list = []
        z_logits_list = []

        optim_latents = init_latents.clone()
        optim_latents = torch.nn.Parameter(optim_latents)

        init_img = self.get_img_from_latents(init_latents, )
        img_size = init_img.shape[2::]
        a = norm.pdf(np.arange(-1, 1, 2 / img_size[0]), 0, 0.2)
        b = a / a.max()
        mask = b[:, None] @ b[None, :]

        mask = torch.tensor(mask).to(self.device, torch.float32)[None, None, :]
        mask[mask < 0.1] = 0

        for zoom_idx in range(num_generations):
            optimizer = torch.optim.AdamW(
                params=[optim_latents],
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=0.1,
            )

            optim_img = self.get_img_from_latents(optim_latents, )

            img_h = optim_img.shape[2]
            img_w = optim_img.shape[3]

            optim_img = torchvision.transforms.functional.affine(
                optim_img,
                angle=0,
                translate=[0, 0],
                scale=1 + zoom_offset / min(img_h, img_w),
                shear=[0, 0],
                resample=PIL.Image.BILINEAR,
            )

            init_img = optim_img.clone().detach()

            zoom_latents = self.get_latents_from_img(
                optim_img,
                num_rec_steps=0,
            )

            optim_latents.data = zoom_latents

            for train_idx in range(num_zoom_train_steps):
                loss = 0

                optim_img = self.get_img_from_latents(optim_latents, )
                # optim_img = mask * optim_img + (1 - mask) * init_img

                optim_img_batch = checkpoint(
                    self.augment(
                        optim_img,
                        num_crops=64,
                        # pad_downscale=8,
                    ))

                loss += 10 * self.compute_clip_loss(
                    img_batch=optim_img_batch,
                    text=prompt,
                )

                # logging.info(
                #     f"Loss {loss} - {train_idx}/{num_zoom_train_steps}")

                # loss += -10 * torch.cosine_similarity(init_latents,
                #                                       optim_latents).mean()

                logging.info(
                    f"Loss {loss} - {train_idx + 1}/{num_zoom_train_steps}")

                def scale_grad(grad, ):
                    grad_size = grad.shape[2:4]
                    grad_mask = torch.nn.functional.interpolate(
                        mask,
                        grad_size,
                        mode="bilinear",
                    )

                    masked_grad = grad * grad_mask

                    return masked_grad

                # optim_img_hook = optim_img.register_hook(scale_grad, )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # optim_img_hook.remove()

            interp_img_list = self.interpolate(
                [init_latents, optim_latents],
                duration_list=[
                    num_zoom_interp_steps / FPS,
                    num_zoom_interp_steps / FPS,
                ],
                interpolation_type="linear",
                loop=False,
            )

            for interp_idx, interp_img in enumerate(interp_img_list):
                print("Adding img...")
                os.makedirs("generations", exist_ok=True)
                interp_img.save(
                    f"generations/{train_idx}_{zoom_idx}_{interp_idx}.jpg")

                interp_img = torchvision.transforms.PILToTensor()(interp_img, )
                interp_img = interp_img.float().to(self.device) / 255.

                gen_img_list.append(interp_img)
                z_logits_list.append(optim_latents.detach().clone(), )

                # NOTE: do not run the last frame
                if interp_idx == len(interp_img_list) - 2:
                    break

            init_latents = optim_latents.detach().clone()

            torch.cuda.empty_cache()

        return gen_img_list, z_logits_list


if __name__ == '__main__':
    target_img_height = 256
    target_img_width = 256
    prompt = "Space mushrooms artsation"
    lr = 0.5
    num_steps = 100
    num_augmentations = 32
    loss_type = 'cosine_similarity'
    init_img_path = None
    # init_img_path = "medusa.jpg"

    save_latents = False
    zoom = False

    clip_model_name_list = [
        "ViT-B/32",
        "ViT-B/16",
        # "RN50x16",
        # "RN50x4",
    ]
    taming_decoder = TamingDecoder(clip_model_name_list=clip_model_name_list, )

    init_latents_path = f"./{prompt}_logits.pt"
    if os.path.exists(init_latents_path) and save_latents:
        init_latents = torch.load(init_latents_path).to('cuda')
    else:

        def save_img_cb(**kwargs, ):
            step = kwargs["step"]
            rec_img = kwargs["rec_img"]

            x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                rec_img[0])
            x_rec_img.save(f"generations/{step}.jpg")

        gen_img_list, z_logits_list = taming_decoder.generate_from_prompt(
            prompt=prompt,
            lr=lr,
            target_img_height=target_img_height,
            target_img_width=target_img_width,
            num_steps=num_steps,
            num_augmentations=num_augmentations,
            init_img_path=init_img_path,
            loss_type=loss_type,
            generation_cb=save_img_cb,
        )

        init_latents = z_logits_list[-1]
        torch.save(init_latents, f'{prompt}_logits.pt')

    if zoom:
        gen_img_list, z_logits_list = taming_decoder.zoom(
            prompt,
            init_latents=init_latents,
        )

    # _gen_img_list, z_logits_list_ = taming_decoder.generate_from_prompt(
    #     prompt="Pokemon of type grass",
    #     lr=lr,
    #     num_steps=num_steps,
    #     num_augmentations=num_augmentations,
    #     init_img_path=init_img_path,
    # )

    # z_logits_interp_list = [z_logits_list[-1], z_logits_list_[-1]]

    # duration_list = [0.7] * len(z_logits_interp_list)
    # interpolate_img_list = taming_decoder.interpolate(
    #     z_logits_list=z_logits_interp_list,
    #     duration_list=duration_list,
    # )
