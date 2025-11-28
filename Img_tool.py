# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 01:49:34 2025

@author: Inaho
"""

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

def mask_sharp(path_in, path_out, radius=1, contrast=1.7, sharpen_radius=1.3, sharpen_percent=180, thresh=140):
    img = Image.open(path_in).convert("RGB")
    arr = np.array(img)
    mask = (arr[:,:,0] < 178) & (arr[:,:,1] < 190) & (arr[:,:,2] < 195)
    binary = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    binary[mask] = 255
    m = Image.fromarray(binary, mode="L")
    m = m.filter(ImageFilter.GaussianBlur(radius=0.6))
    if radius > 0:
        m = m.filter(ImageFilter.MaxFilter(size=2*radius+1))
    m = ImageOps.autocontrast(m, cutoff=1)
    m = ImageEnhance.Contrast(m).enhance(contrast)
    m = m.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=2))
    m = m.point(lambda p: 255 if p >= thresh else 0)
    m.save(path_out)
    return m

mask_sharp("bild_1.jpg", "bild.jpg", radius=1, contrast=1.6, sharpen_radius=1.2, sharpen_percent=180, thresh=140)
