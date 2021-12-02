import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geniverse",
    version="0.0.12",
    author="Javi and Vicc",
    author_email="vipermu97@gmail.com",
    description=
    "Easy library for guiding generative AI models. Find your latent!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thegeniverse/geniverse",
    project_urls={
        "Docs": "https://github.com/thegeniverse/geniverse/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={
    #     "": ".",
    #     "taming": "./geniverse/models/taming/modeling_taming",
    # },
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    # packages=["geniverse", "forks"],
    python_requires=">=3.6",
    install_requires=[
        "tqdm>=4.60.0",
        "clip_by_openai==0.1.1.5",
        "dall-e==0.1",
        "imageio-ffmpeg==0.4.3",
        "PyYAML==5.4.1",
        "omegaconf==2.0.6",
        "pytorch-lightning==1.3.3",
        "einops==0.3.0",
        "imageio==2.9.0",
        "torch==1.7.1",
        "torchvision==0.8.2",
        "tensorboard>=2.2.0",
        "transformers>=4.10.0",
        # "flax>=0.3.4",
        # "jax==0.2.20",

        # "jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
)