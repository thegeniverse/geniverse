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
from omegaconf import OmegaConf
from forks.taming_transformers.taming.models.vqgan import VQModel, GumbelVQ
from geniverse.modeling_utils import ImageGenerator

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
    ) -> None:
        """
        Downloads the VQGAN model and loads a pre-defined 
        configuration.

        Args:
            device (str, optional): defaults to 'cuda'.
        """
        super().__init__()

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
    ):
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

        z, _, [_, _, _indices] = self.vqgan_model.encode(img_tensor)

        z = z.to(self.device)

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
        loss_clip_value: float = None,
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
            loss_clip_value (float, optional): thresholding value for
                for the CLIP embeddings. Defaults to None.
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
                loss_clip_value,
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
        fps = 25

        for idx, (z_logits,
                  duration) in enumerate(zip(z_logits_list, duration_list)):
            num_steps = int(duration * fps)
            z_logits_1 = z_logits
            z_logits_2 = z_logits_list[(idx + 1) % len(z_logits_list)]

            for step in range(num_steps):
                weight = math.sin(1.5708 * step / num_steps)**2
                z_logits = weight * z_logits_2 + (1 - weight) * z_logits_1

                z = self.vqgan_model.post_quant_conv(z_logits)
                rec_img = self.vqgan_model.decoder(z)
                rec_img = (rec_img.clip(-1, 1) + 1) / 2

                x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    rec_img[0])
                gen_img_list.append(x_rec_img)

                torch.cuda.empty_cache()

        return gen_img_list

    def zoom(
        self,
        prompt: str,
        init_latents: torch.Tensor,
        lr: float = 0.008,
        num_generations: int = 64,
        num_zoom_interp_steps=2,
        num_zoom_train_steps=8,
        zoom_offset=4,
    ):
        z_logits = init_latents.detach().clone()
        z_logits = torch.nn.Parameter(z_logits)

        optimizer = torch.optim.AdamW(
            params=[z_logits],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        gen_img_list = []
        z_logits_list = []
        for step in range(num_generations):
            logging.info(f"Generation {step}/{num_generations}")

            with torch.no_grad():
                z = self.vqgan_model.post_quant_conv(z_logits)
                rec_img = self.vqgan_model.decoder(z)
                rec_img = (rec_img.clip(-1, 1) + 1) / 2

                x_rec_size = rec_img.shape[-1]

                x_rec_zoom = rec_img[:, :, zoom_offset:-zoom_offset,
                                     zoom_offset:-zoom_offset]
                x_rec_zoom = torch.nn.functional.interpolate(
                    x_rec_zoom,
                    (x_rec_size, x_rec_size),
                    mode="bilinear",
                )

                x_rec_zoom = 2. * x_rec_zoom - 1
                zoom_z_logits, _, [_, _, indices
                                   ] = self.vqgan_model.encode(x_rec_zoom)

                z_logits.data = zoom_z_logits.clone().detach()

            for zoom_train_step in range(num_zoom_train_steps):
                loss = 0
                z = self.vqgan_model.post_quant_conv(z_logits)
                rec_img = self.vqgan_model.decoder(z)
                rec_img = (rec_img.clip(-1, 1) + 1) / 2
                x_rec_stacked = self.augment(
                    rec_img,
                    rec_img.shape[1],
                    rec_img.shape[2],
                )

                loss += 10 * self.compute_clip_loss(x_rec_stacked, prompt)

                logging.info(
                    f"Loss {loss} - {zoom_train_step}/{num_zoom_train_steps}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            z_logits_1 = init_latents
            z_logits_2 = z_logits

            for zoom_step in range(num_zoom_interp_steps):
                weight = zoom_step / num_zoom_interp_steps
                interp_logits = weight * z_logits_2 + (1 - weight) * z_logits_1

                with torch.no_grad():
                    z = self.vqgan_model.post_quant_conv(interp_logits)
                    rec_img = self.vqgan_model.decoder(z)
                    rec_img = (rec_img.clip(-1, 1) + 1) / 2

                x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    rec_img[0])
                gen_img_list.append(x_rec_img)

                print("Adding img...")
                os.makedirs("generations", exist_ok=True)
                x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    rec_img[0])
                x_rec_img.save(
                    f"generations/{step}_{zoom_step}_{zoom_train_step}.jpg")

            init_latents = z_logits.detach().clone()

            torch.cuda.empty_cache()

        return gen_img_list, z_logits_list


if __name__ == '__main__':
    target_img_height = 256
    target_img_width = 256
    prompt = "A dog"
    lr = 0.1
    num_steps = 1
    num_augmentations = 32
    loss_type = 'cosine_similarity'
    init_img_path = None
    init_img_path = "medusa.jpg"
    loss_clip_value = None

    taming_decoder = TamingDecoder()

    init_latents_path = f"./{prompt}_logits.pt"
    if os.path.exists(init_latents_path):
        init_latents = torch.load(init_latents_path).to('cuda')
    else:

        def save_img_cb(**kwargs, ):
            step = kwargs["step"]
            rec_img = kwargs["rec_img"]

            x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                rec_img[0])
            x_rec_img.save(f"{step}.jpg")

        gen_img_list, z_logits_list = taming_decoder.generate_from_prompt(
            prompt=prompt,
            lr=lr,
            target_img_height=target_img_height,
            target_img_width=target_img_width,
            num_steps=num_steps,
            num_augmentations=num_augmentations,
            init_img_path=init_img_path,
            loss_type=loss_type,
            loss_clip_value=loss_clip_value,
            generation_cb=save_img_cb,
        )

        init_latents = z_logits_list[-1]
        torch.save(init_latents, f'{prompt}_logits.pt')

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
