GENIVERSE DOCS
=======================================================================================================================

State-of-the-art generative image models for Pytorch using CLIP.

-----------------------------------------------------------------------------------------------------------------------

Features
-----------------------------------------------------------------------------------------------------------------------

- ...

Supported models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..
    This list is updated automatically from the README with `make fix-copies`. Do not update manually!

1. :doc:`Taming Decoder <model_doc/taming_decoder>` released in the paper
   `Taming Transformers for High-Resolution Image Synthesis <https://arxiv.org/abs/2012.09841>`__, by 
   Patrick Esser, Robin Rombach, Bjorn Ommer.
2. :doc:`Aphantasia <model_doc/aphantasia>` released by `@eps696 <https://twitter.com/eps696>`__
   in https://github.com/eps696/aphantasia.
3. :doc:`DALL-E Mini <model_doc/dalle_mini>` released in https://github.com/borisdayma/dalle-mini
   by Boris Dayma, Suraj Patil, Pedro Cuenca, Khalid Saifullah, Tanishq Abraham, Phúc Lê, Luke, Ritobrata Ghosh.


Supported features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
    This table is updated automatically from the auto modules with `make fix-copies`. Do not update manually!

.. rst-class:: center-aligned-table

+-----------------------------+----------------------+--------------------+----------------+----------+-----+
|            Model            | Generate from Prompt | Load Initial Image | Interpolations | Zoomings | ... |
+=============================+======================+====================+================+==========+=====+
|       Taming Decoder        |          ✅          |         ✅         |       ✅       |    ❌    | ❌  |
+-----------------------------+----------------------+--------------------+----------------+----------+-----+
|         Aphantasia          |          ✅          |         ❌         |       ✅       |    ❌    | ❌  |
+-----------------------------+----------------------+--------------------+----------------+----------+-----+
|         DALL-E Mini         |          ✅          |         ❌         |       ❌       |    ❌    | ❌  |
+-----------------------------+----------------------+--------------------+----------------+----------+-----+

.. toctree::
    :maxdepth: 2
    :caption: Get started

    installation

.. toctree::
    :maxdepth: 1
    :caption: Models

    model_doc/taming_decoder
    model_doc/aphantasia
    model_doc/dalle_mini

