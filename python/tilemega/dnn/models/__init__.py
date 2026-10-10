"""Import the unchanged NAFNet architecture with its upstream dependencies."""
import sys
from pathlib import Path


def nafnet():
    upstream = str(Path(__file__).with_name('_upstream'))
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from .nafnet_arch import NAFNet
    return NAFNet(img_channel=3, width=32, middle_blk_num=12,
                  enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
