import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("envs/geniverse-requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().strip().split("\n")

setuptools.setup(
    name="geniverse",
    version="0.1.0",
    author="Geniverse",
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
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requirements,
)
