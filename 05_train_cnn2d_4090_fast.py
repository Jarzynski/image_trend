# -*- coding: utf-8 -*-
"""
Compatibility entrypoint for the optimized CNN trainer.

The optimized 4080/4090 implementation is now the official 05 script:
    05_train_cnn2d.py

This wrapper is kept so existing cluster scripts that call
`python 05_train_cnn2d_4090_fast.py` continue to work without maintaining a
second copy of the CNN training logic.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("05_train_cnn2d.py")), run_name="__main__")
