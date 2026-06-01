# -*- coding: utf-8 -*-
"""
Compatibility entrypoint for the optimized image generator.

The fast implementation is now the official 03 script:
    03_make_images.py

This wrapper is kept so existing cluster scripts that call
`python 03_make_images_fast.py` continue to work without maintaining a second
copy of the image-generation logic.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("03_make_images.py")), run_name="__main__")
