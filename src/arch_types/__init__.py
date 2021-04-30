import glob
import os

import yaml

SRC_DIR = __name__.replace('.', '/')
ARCH_TYPES = [os.path.basename(file).replace(".yaml", "") for file in glob.glob(f"{SRC_DIR}/*.yaml")]


def get_arch_cells(arch):
    if arch in ARCH_TYPES:
        return yaml.load(open(f"{SRC_DIR}/{arch}.yaml").read(), Loader=yaml.CLoader)
    raise NotImplementedError()
