import config
from cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:#False
    enable_sliced_attention()