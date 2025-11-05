import cv2
import numpy as np

HEADER_BITS = 32

# mid-band mask (avoid DC & very high freq)
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

def _str_to_bits(s):
    return [int(x) for x in ''.join(format(ord(c),'08b') for c in s)]

def _bits_to_str(bits):
    out = []
    for i in range(0, len(bits), 8):
        b = bits[i:i+8]
        if len(b) < 8: break
        out.append(chr(int(''.join(map(str,b)),2)))
    return ''.join(out)

def _int_to_bits32(n): return [int(x) for x in format(n,'032b')]
def _bits32_to_int(bits): return int(''.join(map(str,bits[:32])),2)

def _block_process(Y, func):
    H, W = Y.shape
    H8, W8 = H - (H % 8), W - (W % 8)
    out = Y.copy()
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            block = Y[i:i+8, j:j+8].astype(np.float32) - 128.0
            dct = cv2.dct(block)
            dct2 = func(dct)
            idct = cv2.idct(dct2) + 128.0
            out[i:i+8, j:j+8] = np.clip(idct, 0, 255)
    return out

def embed_message(cover_image_path, message, output_path):
    img = cv2.imread(cover_image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not read cover image.")
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0].astype(np.float32)

    bits = _str_to_bits(message)
    header = _int_to_bits32(len(bits))
    payload = header + bits
    bit_idx = 0
    total_slots = ((Y.shape[0]//8)*(Y.shape[1]//8)) * int(MID_MASK.sum())
    if len(payload) > total_slots:
        raise ValueError(f"Message too large for DCT. Capacity={total_slots} bits; need={len(payload)}")

    def embed_block(dct):
        nonlocal bit_idx
        d = dct.copy()
        # iterate through mid-band positions
        for u in range(8):
            for v in range(8):
                if MID_MASK[u,v] and bit_idx < len(payload):
                    coeff = d[u,v]
                    # Set LSB of rounded coefficient to target bit (on absolute integer)
                    ci = int(np.round(coeff))
                    if (ci & 1) != payload[bit_idx]:
                        ci += 1 if ci % 2 == 0 else -1
                    d[u,v] = float(ci)
                    bit_idx += 1
        return d

    Y2 = _block_process(Y, embed_block).astype(np.uint8)
    YCrCb[:,:,0] = Y2
    stego = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    if not cv2.imwrite(output_path, stego):
        raise ValueError("Failed to write stego image.")

def extract_message(stego_path):
    img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not read stego image.")
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0].astype(np.float32)

    bits = []

    def extract_block(dct):
        nonlocal bits
        for u in range(8):
            for v in range(8):
                if MID_MASK[u,v]:
                    ci = int(np.round(dct[u,v]))
                    bits.append(ci & 1)
        return dct

    _ = _block_process(Y, extract_block)  # only to iterate
    msg_len = _bits32_to_int(bits[:HEADER_BITS])
    total = HEADER_BITS + msg_len
    if total > len(bits):
        raise ValueError("Declared message length exceeds extracted bits.")
    payload = bits[HEADER_BITS:total]
    return _bits_to_str(payload)
