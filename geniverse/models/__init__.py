import logging

try:
    from .taming.modeling_taming import TamingDecoder
except Exception as e:
    logging.warning(repr(e))
    logging.warning("WARNING! TAMING NOT AVAILABLE.")
try:
    from .aphantasia.modeling_aphantasia import Aphantasia
except Exception as e:
    logging.warning(repr(e))
    logging.warning("WARNING! APHANTASIA NOT AVAILABLE.")
try:
    from .dalle_mini.modeling_dalle_mini import DalleMini
except Exception as e:
    logging.warning(repr(e))
    logging.warning("WARNING! DALLE MINI NOT AVAILABLE.")