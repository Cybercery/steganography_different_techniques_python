# backend/algorithms/hybrid_dwt_dct.py
import cv2
import numpy as np
import pywt

HEADER_BITS = 32

MID_MASK = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0],
    [0,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0]
], dtype=np.uint8)

def _str_to_bits(s): return [int(x) for x in ''.join(format(ord(c),'08b') for c in s)]
def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8: break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)
def _int_to_bits32(n): return [int(x) for x in format(n,'032b')]
def _bits32_to_int(bits): return int(''.join(map(str,bits[:32])),2)

def _crop_all_equal8(LL, LH, HL, HH):
    """Crop all subbands to the SAME size that's a multiple of 8 in both dims."""
    h = min(LL.shape[0], LH.shape[0], HL.shape[0], HH.shape[0])
    w = min(LL.shape[1], LH.shape[1], HL.shape[1], HH.shape[1])
    h8 = h - (h % 8)
    w8 = w - (w % 8)
    if h8 == 0 or w8 == 0:
        raise ValueError("Image too small for 8x8 hybrid embedding after DWT.")
    return (LL[:h8, :w8].astype(np.float32),
            LH[:h8, :w8].astype(np.float32),
            HL[:h8, :w8].astype(np.float32),
            HH[:h8, :w8].astype(np.float32))

def embed_message(cover_image_path, message, output_path):
    img = cv2.imread(cover_image_path)
    if img is None:
        raise ValueError("Image not found.")
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0].astype(np.float32)

    # DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')

    # Crop ALL bands to equal shape and multiples of 8
    LLc, LHc, HLc, HHc = _crop_all_equal8(LL, LH, HL, HH)

    bits = _str_to_bits(message)
    header = _int_to_bits32(len(bits))
    payload = header + bits

    h8, w8 = HLc.shape
    # capacity = blocks * midband slots
    capacity = (h8 * w8 // 64) * int(MID_MASK.sum())
    if len(payload) > capacity:
        raise ValueError(f"Message too large for Hybrid. Capacity={capacity}, need={len(payload)}")

    bit_idx = 0
    # DCT on 8x8 blocks of HL
    HLw = HLc.copy()
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            if bit_idx >= len(payload):
                break
            block = HLw[i:i+8, j:j+8]
            dct = cv2.dct(block)
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u, v] and bit_idx < len(payload):
                        c = int(np.round(dct[u, v]))
                        if (c & 1) != payload[bit_idx]:
                            c += 1 if (c % 2) == 0 else -1
                        dct[u, v] = float(c)
                        bit_idx += 1
            HLw[i:i+8, j:j+8] = cv2.idct(dct)

    # Inverse DWT with cropped, equal-sized bands
    Y2 = pywt.idwt2((LLc, (LHc, HLw, HHc)), 'haar')

    # Resize back to original Y size to avoid broadcast issues
    Y2 = cv2.resize(Y2, (Y.shape[1], Y.shape[0]))
    YCrCb[:, :, 0] = np.clip(Y2, 0, 255).astype(np.uint8)
    stego = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, stego)

def extract_message(stego_path):
    img = cv2.imread(stego_path)
    if img is None:
        raise ValueError("Could not open stego image.")
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0].astype(np.float32)

    # DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')

    # Crop ALL bands identically as in embed
    LLc, LHc, HLc, HHc = _crop_all_equal8(LL, LH, HL, HH)

    h8, w8 = HLc.shape
    bits = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = HLc[i:i+8, j:j+8]
            dct = cv2.dct(block.astype(np.float32))
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u, v]:
                        bits.append(int(np.round(dct[u, v])) & 1)

    msg_len = _bits32_to_int(bits[:HEADER_BITS])
    total = HEADER_BITS + msg_len
    if total > len(bits):
        raise ValueError("Declared length exceeds extracted bits (corrupted stego).")
    payload = bits[HEADER_BITS:total]
    return _bits_to_str(payload)
