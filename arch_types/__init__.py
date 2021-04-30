import glob
import os

import yaml

ARCH_TYPES = [os.path.basename(file).replace(".yaml", "") for file in glob.glob(f"{__name__}/*.yaml")]


def get_arch_cells(arch):
    if arch in ARCH_TYPES:
        return yaml.load(open(f"{__name__}/{arch}.yaml").read(), Loader=yaml.CLoader)
    raise NotImplementedError()
