import os
import math
import logging
from typing import *

import torch
import torchvision
import PIL
from PIL import Image
import numpy as np
from geniverse.modeling_utils import ImageGenerator

logging.basicConfig(format='%(message)s', level=logging.INFO)

NUM_IMG_CHANNELS = 3
NUM_IMAGINARY_CHANNELS = 2


class Aphantasia(ImageGenerator):
    """
    Image generator that leverages Fourier Transforms to optimize
    images using CLIP. This code uses a fork of the original
    implementation from https://github.com/eps696/aphantasia.
    """
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__(device=device, )

        if device is not None:
            self.device = device

        modeling_dir = os.path.dirname(os.path.abspath(__file__))
        modeling_cache_dir = os.path.join(modeling_dir, ".modeling_cache")
        os.makedirs(modeling_cache_dir, exist_ok=True)

    # From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    @staticmethod
    def compute_2d_img_freqs(
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Computes a 2D frequency matrix.

        Args:
            height (int): height.
            width (int): width.

        Returns:
            np.ndarray: frequency image.
        """
        y_freqs = np.fft.fftfreq(height)[:, None]

        # NOTE: when we have an odd input dimension we need to keep one
        # additional frequency and later cut off 1 pixel
        width_even_idx = (width + 1) // 2 if width % 2 == 1 else width // 2 + 1

        x_freqs = np.fft.fftfreq(width)[:width_even_idx]

        img_freqs = np.sqrt(x_freqs * x_freqs + y_freqs * y_freqs)

        return img_freqs

    @staticmethod
    def get_scale_from_img_freqs(
        img_freqs: np.ndarray,
        decay_power: float,
    ) -> torch.Tensor:
        """
        Given a numpy array representing the frequencies of an image
        this function computes a matrix with weights in order to 
        scale each of these frequencies when transforming the image
        to RGB. This is required due to the non-linear importance of
        each frequency.

        Args:
            img_freqs (np.ndarray): input frequency images.
            decay_power (float): controls the amount of decay in the
                computed frequencies.

        Returns:
            torch.Tensor: scaling matrix for the frequency image
                provided.
        """
        height, width = img_freqs.shape
        clamped_img_freqs = np.maximum(img_freqs, 1.0 / max(width, height))

        scale = 1.0 / clamped_img_freqs**decay_power
        scale *= np.sqrt(width * height)
        scale = torch.tensor(scale).float()[None, None, ..., None]

        return scale

    def fft_to_rgb(
        self,
        fft_img: torch.Tensor,
        scale: float,
        height: int,
        width: int,
        shift: float = None,
        contrast: float = 1.,
        decorrelate: bool = True,
    ) -> torch.Tensor:
        """
        Transforms an image from frequency domain to RGB space using
        the inverse fourier transform.

        Args:
            fft_img (torch.Tensor): image in frequency domain to be
                transformed.
            scale (float): scale of each of the frequencies in the 
                input frequency image.
            height (int): height of the final RGB image.
            width (int): width of the final RGB image.
            shift (float, optional): if not None this value gets
                multiplied to the scale and the result is added to 
                the frequency image (hence producing a shift on its
                values). Useful if we want to center low frequencies
                for example. Defaults to None.
            contrast (float, optional): this value multiplies the 
                resulting images. Empirically it regulates the amount
                of contrast in the final RGB image. Defaults to 1.
            decorrelate (bool, optional): if True applies a color 
                transformation to the RGB images. It is recommended
                since it makes the final image more vivid. Defaults 
                to True.

        Returns:
            torch.Tensor: tensor representing the transformed RGB 
                image.
        """
        scaled_fft_img = scale * fft_img
        if shift is not None:
            scaled_fft_img += scale * shift

        img_size = (height, width)

        image = torch.irfft(
            scaled_fft_img,
            NUM_IMAGINARY_CHANNELS,
            normalized=True,
            signal_sizes=img_size,
        )
        image = image * contrast / image.std()  # keep contrast, empirical

        if decorrelate:
            colors = 1
            color_correlation_svd_sqrt = np.asarray([
                [0.26, 0.09, 0.02],
                [0.27, 0.00, -0.05],
                [0.27, -0.09, 0.03],
            ]).astype("float32")
            color_correlation_svd_sqrt /= np.asarray([
                colors,
                1.,
                1.,
            ])  # saturate, empirical

            max_norm_svd_sqrt = np.max(
                np.linalg.norm(color_correlation_svd_sqrt, axis=0))

            color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

            image_permute = image.permute(0, 2, 3, 1)
            image_permute = torch.matmul(
                image_permute,
                torch.tensor(color_correlation_normalized.T).to(self.device))

            image = image_permute.permute(0, 3, 1, 2)

        image = torch.sigmoid(image)

        return image

    def get_random_latents(
        self,
        target_img_height: int,
        target_img_width: int,
        batch_size: int = 1,
        std: float = 0.01,
    ) -> torch.Tensor:
        """
        This function produces a tensor of size `width` x `height`
        containing random fft values. If `return_img_freqs` is True
        it also return the image frequencies.

        Args:
            target_img_height (int): height of the final image.
            width (int): width of the final image.
            batch_size (int, optional): batch size of the resulting 
                tensor. Defaults to 1.
            std (float, optional): standard deviation of the normal 
                distribution used to sample the random samples for 
                the fft image. Defaults to 0.01.
            return_img_freqs (bool, optional): if True the function
                also returns all the possible frequencies within the
                space of the image. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: either random 
                tensor of fft images or Tuple of the random fft 
                images and its frequency space.
        """
        #NOTE: generat all possible freqs for the input image size
        img_freqs = self.compute_2d_img_freqs(
            height=target_img_height,
            width=target_img_width,
        )

        spectrum_shape = [
            batch_size,
            NUM_IMG_CHANNELS,
            *img_freqs.shape,
            NUM_IMAGINARY_CHANNELS,
        ]

        fft_img = (torch.randn(*spectrum_shape) * std)

        return fft_img

    def get_img_size_from_latents(
        self,
        latents,
    ):
        height = latents.shape[2]
        width = (latents.shape[3] - 1) * 2

        return height, width

    def get_img_from_latents(
        self,
        latents,
        target_img_height=None,
        target_img_width=None,
    ):
        shift = None

        if target_img_height is None or target_img_width is None:
            target_img_height, target_img_width = self.get_img_size_from_latents(
                latents, )

        #NOTE: generat all possible freqs for the input image size
        img_freqs = self.compute_2d_img_freqs(
            height=target_img_height,
            width=target_img_width,
        )

        scale = self.get_scale_from_img_freqs(
            img_freqs=img_freqs,
            decay_power=1,
        ).to(self.device)

        img = self.fft_to_rgb(
            fft_img=latents,
            scale=scale,
            height=target_img_height,
            width=target_img_width,
            shift=shift,
            contrast=1.0,
            decorrelate=True,
        )

        return img

    def generate_from_prompt(
        self,
        prompt: str,
        lr: float = 3e-1,
        num_steps: int = 200,
        num_random_crops: int = 20,
        height: int = 256,
        width: int = 256,
        generation_cb=None,
        init_embed: torch.Tensor = None,
        loss_type: str = "cosine_similarity",
        **kwargs,
    ) -> Tuple[List[PIL.Image.Image], List[torch.Tensor]]:
        """
        Returns a list of images generated while optimizing
        fft coefficients.

        Args:
            prompt (str): input prompt.
            lr (float, optional): learning rate. Defaults to 0.5.
            num_steps (int, optional): number of optimization 
                steps. Defaults to 200.
            num_random_crops (int, optional): number of augmented 
                images to use. Defaults to 20.
            height (int, optional): height of the optimized image. 
                Defaults to 256.
            width (int, optional): wifth of the optimized image. 
                Defaults to 256.
            generation_cb (function, optional): if provided, the 
                function is executed for every optimization step. The
                inputs of this function will be the generated image
                from the VQGAN (Nx3xHxW) and the logits used to 
                generate it (Nx256xDxD) and the optimization step.
            init_embed (torh.Tensor, optional): initial embedding 
                From where to start the generation. Defaults to None.

        Returns:
            Tuple[List[PIL.Image.Image], List[torch.Tensor]]: list 
                of optimized images and their respective fft
                encodings.
        """
        # TODO: implement batching
        batch_size = 1

        if loss_type not in self.supported_loss_types:
            print(
                (f"ERROR! Loss type {loss_type} not recognized. "
                 f"Only {' or '.join(self.supported_loss_types)} supported."))

            return

        if init_embed is not None:
            fft_img = init_embed

        else:
            fft_img = self.get_random_latents(
                height,
                width,
                batch_size=1,
                std=0.01,
            )

        fft_img = fft_img.to(self.device)
        fft_img.requires_grad = True
        fft_img = torch.nn.Parameter(fft_img)

        # if noise > 0:
        #     img_size = img_freqs.shape
        #     noise_size = (1, 1, *img_size, 1)
        #     shift = self.noise * torch.randn(noise_size, ).to(self.device)

        optimizer = torch.optim.Adam(
            [fft_img],
            lr,
        )

        # NOTE: with SGD the results are less complex but still good
        # optimizer = torch.optim.SGD(
        #     [fft_img],
        #     lr,
        # )

        gen_img_list = []
        gen_fft_list = []
        for step in range(num_steps):
            loss = 0

            initial_img = self.get_img_from_latents(latents=fft_img, )

            x_rec_stacked = self.augment(
                img_batch=initial_img,
                num_crops=num_random_crops,
                target_img_height=height,
                target_img_width=width,
            )
            loss += 10 * self.compute_clip_loss(x_rec_stacked, prompt)

            logging.info(f"\nIteration {step} of {num_steps}")
            logging.info(f"Loss {round(float(loss.data), 2)}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rec_img = self.get_img_from_latents(latents=fft_img, )
                x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    rec_img[0], )

                gen_img_list.append(x_rec_img)
                gen_fft_list.append(fft_img)

                if generation_cb is not None:
                    generation_cb(
                        loss=loss,
                        step=step,
                        rec_img=rec_img,
                        latents=fft_img,
                    )

            torch.cuda.empty_cache()

        return gen_img_list, gen_fft_list

    def interpolate(
        self,
        latents_list: List,
        duration_list: List[float],
        height: int = None,
        width: int = None,
        **kwargs,
    ):
        """
        Returns a list of PIL images corresponding to the 
        generations produced by interpolating the values
        from `latents_list`.

        Args:
            latents_list (List[torch.Tensor]): list of fft tensors
                that encode image generations.
            duration_list (List[float]): list with the duration of
                the interpolation between each consecutive tensor. 
                The last value represent the duration between the 
                last tensor and the initial.
            height (int, optional): height of the optimized image. 
                Defaults to 256.
            wifth (int, optional): wifth of the optimized image. 
                Defaults to 256.

        Returns:
            List[PIL.Image.Image]: list of the resulting generated images.
        """
        fft_img_list = latents_list

        gen_img_list = []
        fps = 25

        if height is None or width is None:
            logging.info("Inferring image size")
            height, width = self.get_img_size_from_latents(latents_list[0], )

        for idx, (fft_img,
                  duration) in enumerate(zip(fft_img_list, duration_list)):
            num_steps = int(duration * fps)
            fft_img_1 = fft_img
            fft_img_2 = fft_img_list[(idx + 1) % len(fft_img_list)]

            for step in range(num_steps):
                weight = math.sin(1.5708 * step / num_steps)**2
                fft_img_interp = weight * fft_img_2 + (1 - weight) * fft_img_1
                img = self.get_img_from_latents(latents=fft_img_interp, )

                img = img.detach().cpu().numpy()[0]
                img = np.transpose(np.array(img)[:, :, :], (1, 2, 0))
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)

                gen_img_list.append(pil_img)

                torch.cuda.empty_cache()

        return gen_img_list


if __name__ == '__main__':
    prompt = "'Tempus Fugit' by Salvador Dali"
    lr = 0.5
    num_steps = 64
    num_random_crops = 32
    height = 256
    width = 512

    def save_img_cb(**kwargs, ):
        step = kwargs["step"]
        rec_img = kwargs["rec_img"]

        x_rec_img = torchvision.transforms.ToPILImage(mode='RGB')(rec_img[0])
        x_rec_img.save(f"{step}.jpg")

    aphantasia = Aphantasia()
    gen_img_list, fft_logits_list = aphantasia.generate_from_prompt(
        prompt=prompt,
        lr=lr,
        num_steps=num_steps,
        num_random_crops=num_random_crops,
        height=height,
        width=width,
        generation_cb=save_img_cb,
    )

    _gen_img_list, fft_logits_list_ = aphantasia.generate_from_prompt(
        prompt="'Tempus Fugit', by Escher",
        lr=lr,
        num_steps=num_steps,
        num_random_crops=num_random_crops,
        height=height,
        width=width,
    )

    fft_logits_interp_list = [fft_logits_list[-1], fft_logits_list_[-1]]

    duration_list = [0.7] * len(fft_logits_interp_list)
    interpolate_img_list = aphantasia.interpolate(
        fft_logits_list=fft_logits_interp_list,
        duration_list=duration_list,
        height=height,
        width=width,
    )
