import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import os

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

def mse(img1, img2):
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0: return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(m)))

def _image_metrics(orig_path, stego_path):
    o = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    s = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if o is None or s is None: 
        return {'MSE': None, 'PSNR': None, 'SSIM': None}
    # resize to match if needed (safety)
    if o.shape != s.shape:
        s = cv2.resize(s, (o.shape[1], o.shape[0]))
    m = mse(o, s)
    p = psnr(o, s)
    so = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    ss = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    si = float(ssim(so, ss))
    return {'MSE': m, 'PSNR': p, 'SSIM': si}

def _video_metrics(orig_path, stego_path, max_frames=10):
    co = cv2.VideoCapture(orig_path)
    cs = cv2.VideoCapture(stego_path)
    if not (co.isOpened() and cs.isOpened()):
        return {'MSE': None, 'PSNR': None, 'SSIM': None}
    frames = 0
    mses, psnrs, ssims = [], [], []
    while frames < max_frames:
        ro, o = co.read()
        rs, s = cs.read()
        if not (ro and rs): break
        if o.shape != s.shape:
            s = cv2.resize(s, (o.shape[1], o.shape[0]))
        mses.append(mse(o, s))
        psnrs.append(psnr(o, s))
        so = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
        ss = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        ssims.append(ssim(so, ss))
        frames += 1

    # MSE full form: Mean Squared Error which measures the average squared difference between pixel values of the original and stego images.
    # It quantifies the overall distortion introduced by the steganography process.
    # lower MSE values indicate better quality.

    # PSNR full form: Peak Signal-to-Noise Ratio is a logarithmic measure that
    # compares the maximum possible pixel value to the noise introduced by steganography. Higher PSNR values indicate better quality.
    # lower PSNR values indicate better quality.

    # SSIM full form: Structural Similarity Index Measure evaluates the perceptual similarity between the original and stego images. 
    # It considers changes in structural information, luminance, and contrast, providing a more human-centric assessment of image quality.
    # SSIM values range from -1 to 1, with higher values indicating greater similarity.


    co.release(); cs.release()
    if frames == 0:
        return {'MSE': None, 'PSNR': None, 'SSIM': None}
    return {
        'MSE': float(np.mean(mses)),
        'PSNR': float(np.mean(psnrs)),
        'SSIM': float(np.mean(ssims))
    }

def get_metrics(original, stego, start_time, end_time):
    ext = os.path.splitext(original)[1].lower()
    if ext in IMAGE_EXTS:
        base = _image_metrics(original, stego)
    elif ext in VIDEO_EXTS:
        base = _video_metrics(original, stego)
    else:
        base = {'MSE': None, 'PSNR': None, 'SSIM': None}
    base['Time'] = float(end_time - start_time)
    return base
