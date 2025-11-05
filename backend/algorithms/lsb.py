import cv2
import numpy as np

HEADER_BITS = 32

def _str_to_bits(s):
    return [int(x) for x in ''.join(format(ord(c), '08b') for c in s)]

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8: break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def _int_to_bits32(n): return [int(x) for x in format(n, '032b')]
def _bits32_to_int(bits): return int(''.join(map(str, bits[:32])), 2)

def embed_message(cover_image_path, message, output_path):
    img = cv2.imread(cover_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not open image.")

    msg_bits = _str_to_bits(message)
    header = _int_to_bits32(len(msg_bits))
    payload = header + msg_bits

    flat = img.flatten().astype(np.int32)
    if len(payload) > flat.size:
        raise ValueError("Message too large for this image.")

    flat[:len(payload)] = (flat[:len(payload)] & ~1) | np.array(payload, dtype=np.int32)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    stego = flat.reshape(img.shape)
    cv2.imwrite(output_path, stego)

def extract_message(stego_path):
    img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not open stego image.")

    flat = img.flatten()
    header_bits = flat[:HEADER_BITS] & 1
    msg_len = _bits32_to_int(header_bits)
    msg_bits = flat[HEADER_BITS:HEADER_BITS+msg_len] & 1
    return _bits_to_str(msg_bits.tolist())
