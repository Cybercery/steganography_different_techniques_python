import cv2
import numpy as np
import pywt

HEADER_BITS = 32
STRENGTH = 4  # Embedding strength

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
    return [int(x) for x in ''.join(format(ord(c), '08b') for c in s)]

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8: break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def _int_to_bits32(n): 
    return [int(x) for x in format(n, '032b')]

def _bits32_to_int(bits): 
    return int(''.join(map(str, bits[:32])), 2)


def _crop_all_equal8(LL, LH, HL, HH):
    """Crop subbands to same size that's a multiple of 8."""
    h = min(LL.shape[0], LH.shape[0], HL.shape[0], HH.shape[0])
    w = min(LL.shape[1], LH.shape[1], HL.shape[1], HH.shape[1])
    h8 = h - (h % 8)
    w8 = w - (w % 8)
    return (
        LL[:h8, :w8].astype(np.float32),
        LH[:h8, :w8].astype(np.float32),
        HL[:h8, :w8].astype(np.float32),
        HH[:h8, :w8].astype(np.float32),
    )


def embed_message(cover_image_path, message, output_path, strength=STRENGTH):
    img = cv2.imread(cover_image_path)
    if img is None:
        raise ValueError("Image not found.")

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0].astype(np.float32)
    orig_shape = Y.shape

    # DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')
    LLc, LHc, HLc, HHc = _crop_all_equal8(LL, LH, HL, HH)

    bits = _str_to_bits(message)
    header = _int_to_bits32(len(bits))
    payload = header + bits

    h8, w8 = HLc.shape
    capacity = (h8 * w8 // 64) * int(MID_MASK.sum())
    if len(payload) > capacity:
        raise ValueError(f"Message too large for Hybrid. Capacity={capacity}, need={len(payload)}")

    bit_idx = 0
    HLw = HLc.copy()

    # Embed using quantization in DCT of DWT coefficients
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            if bit_idx >= len(payload):
                break
            block = HLw[i:i+8, j:j+8]
            dct = cv2.dct(block)
            
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u, v] and bit_idx < len(payload):
                        coeff = dct[u, v]
                        target_bit = payload[bit_idx]
                        
                        # Quantization-based embedding
                        quantized = round(coeff / (strength * 2)) * (strength * 2)
                        new_coeff = quantized + (target_bit * strength) + (strength // 2)
                        
                        dct[u, v] = new_coeff
                        bit_idx += 1
            
            HLw[i:i+8, j:j+8] = cv2.idct(dct)

    # Pad subbands back to original shape before inverse DWT
    pad_h = LL.shape[0] - LLc.shape[0]
    pad_w = LL.shape[1] - LLc.shape[1]
    if pad_h or pad_w:
        LLc = np.pad(LLc, ((0, pad_h), (0, pad_w)), 'edge')
        LHc = np.pad(LHc, ((0, pad_h), (0, pad_w)), 'edge')
        HLw = np.pad(HLw, ((0, pad_h), (0, pad_w)), 'edge')
        HHc = np.pad(HHc, ((0, pad_h), (0, pad_w)), 'edge')

    # Reconstruct Y2 with idwt2
    Y2 = pywt.idwt2((LLc, (LHc, HLw, HHc)), 'haar')

    # Clip to match original shape precisely
    Y2 = Y2[:orig_shape[0], :orig_shape[1]]
    YCrCb[:, :, 0] = np.clip(Y2, 0, 255).astype(np.uint8)
    stego = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    
    # Save with no compression
    cv2.imwrite(output_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Embedded {len(payload)} bits ({len(message)} chars) with strength={strength}")


def extract_message(stego_path, strength=STRENGTH):
    img = cv2.imread(stego_path)
    if img is None:
        raise ValueError("Could not open stego image.")

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0].astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')
    LLc, LHc, HLc, HHc = _crop_all_equal8(LL, LH, HL, HH)

    h8, w8 = HLc.shape
    bits = []
    
    # Extract using quantization
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = HLc[i:i+8, j:j+8]
            dct = cv2.dct(block)
            
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u, v]:
                        coeff = dct[u, v]
                        # Extract bit based on quantization remainder
                        remainder = round(coeff) % (strength * 2)
                        bit = 1 if remainder >= strength else 0
                        bits.append(bit)

    if len(bits) < HEADER_BITS:
        raise ValueError("Not enough bits extracted.")
    
    msg_len = _bits32_to_int(bits[:HEADER_BITS])
    print(f"Header says message length: {msg_len} bits")
    
    if HEADER_BITS + msg_len > len(bits):
        raise ValueError(f"Message length {msg_len} exceeds capacity.")
    
    payload = bits[HEADER_BITS:HEADER_BITS + msg_len]
    print(f"Extracted {len(payload)} bits")
    print(f"First 64 bits: {payload[:64]}")
    
    return _bits_to_str(payload)