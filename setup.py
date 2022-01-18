import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geniverse",
    version="0.1.9",
    author="Geniverse",
    author_email="vipermu97@gmail.com",
    description="Open-source library for guiding generative AI models.",
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
    install_requires=["geniverse-hub>=0.0.5"],
)
