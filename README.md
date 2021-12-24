# The Geniverse API
Open source library to access and utilize generative AI models. 

## Install
```bash
pip install geniverse
```

## Usage
With this example we can generate `"a genierative universe"` using `VQGAN`. The result is a list of PIL images. Docs will be released soon :)
```python
from geniverse.models import TamingDecoder
taming_decoder = TamingDecoder()
image_list = taming_decoder.generate_from_prompt("a generative universe")
```

For now you can generate images from text prompts, interpolate and create infinite zooms with your results. More examples soon :)

## Models

So far we have adapted to our pipeline the following models:
- VQGAN
- Aphantasia
- Mini-Dalle

