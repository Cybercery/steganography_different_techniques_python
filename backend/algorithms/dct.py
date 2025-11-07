import cv2
import numpy as np

HEADER_BITS = 32  # 32-bit header stores payload length in BITS
STRENGTH = 4  # Embedding strength: larger => more robust but larger distortion

# 8x8 mid-band mask: choose coefficients that are not DC (0,0) and not extreme high-freq.
# This is a common mid-band selection for 8x8 DCT steganography.
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
    # Convert a Python string to list of bits
    return [int(x) for x in ''.join(format(ord(c),'08b') for c in s)]

def _bits_to_str(bits):
    # Convert list of bits back to string. Stops on incomplete trailing byte.
    out = []
    for i in range(0, len(bits), 8):
        b = bits[i:i+8]
        if len(b) < 8: break
        out.append(chr(int(''.join(map(str,b)),2)))
    return ''.join(out)

def _int_to_bits32(n): 
    # Represent integer n as 32-bit list (MSB first).
    return [int(x) for x in format(n,'032b')]
def _bits32_to_int(bits): return int(''.join(map(str,bits[:32])),2)

def embed_message(cover_image_path, message, output_path, strength=STRENGTH):
    # Read the image (BGR). We use IMREAD_COLOR to ensure 3 channels.
    img = cv2.imread(cover_image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not read cover image.")
    # Convert to YCrCb and operate on Y (luma) channel only.
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0].astype(np.float32)

    #Build bit payload: 32-bit header + message bits
    bits = _str_to_bits(message)
    header = _int_to_bits32(len(bits))
    payload = header + bits #list of 0/1
    
    H, W = Y.shape
    # Crop to nearest multiple of 8
    H8, W8 = H - (H % 8), W - (W % 8)
    # total available slots
    total_slots = ((H8//8)*(W8//8)) * int(MID_MASK.sum())
    
    if len(payload) > total_slots:
        raise ValueError(f"Message too large. Capacity={total_slots} bits; need={len(payload)}")

    bit_idx = 0
    Y2 = Y.copy()
    
    # Process each 8x8 block
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            # Extract 8x8 block and apply DCT
            block = Y[i:i+8, j:j+8].astype(np.float32) - 128.0
            dct = cv2.dct(block)
            
            # Embed bits with stronger modification
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u,v] and bit_idx < len(payload):
                        coeff = dct[u,v]
                        target_bit = payload[bit_idx]
                        
                        # Quantize to multiple of strength*2, then add bit
                        quantized = round(coeff / (strength * 2)) * (strength * 2)
                        new_coeff = quantized + (target_bit * strength) + (strength // 2)
                        
                        dct[u,v] = new_coeff
                        bit_idx += 1
            
            # Inverse DCT
            idct = cv2.idct(dct) + 128.0
            Y2[i:i+8, j:j+8] = np.clip(idct, 0, 255)
    
    YCrCb[:,:,0] = Y2.astype(np.uint8)
    stego = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    
    # Save with maximum quality PNG
    if not cv2.imwrite(output_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
        raise ValueError("Failed to write stego image.")
    
    print(f"Embedded {len(payload)} bits ({len(message)} chars) with strength={strength}")
    return len(payload)

def extract_message(stego_path, strength=STRENGTH):
    img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not read stego image.")
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0].astype(np.float32)

    bits = []
    H, W = Y.shape
    H8, W8 = H - (H % 8), W - (W% 8)
    
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            block = Y[i:i+8, j:j+8].astype(np.float32) - 128.0
            dct = cv2.dct(block)
            
            # Extract bits
            for u in range(8):
                for v in range(8):
                    if MID_MASK[u,v]:
                        coeff = dct[u,v]
                        # Extract bit based on quantization
                        remainder = round(coeff) % (strength * 2)
                        bit = 1 if remainder >= strength else 0
                        bits.append(bit)
    
    if len(bits) < HEADER_BITS:
        raise ValueError("Not enough bits extracted.")
    
    msg_len = _bits32_to_int(bits[:HEADER_BITS])
    print(f"Header says message length: {msg_len} bits")
    
    total = HEADER_BITS + msg_len
    
    if total > len(bits):
        raise ValueError(f"Declared message length {msg_len} exceeds extracted bits {len(bits)-HEADER_BITS}.")
    
    payload = bits[HEADER_BITS:total]
    
    print(f"Extracted {len(payload)} bits")
    print(f"First 64 bits: {payload[:64]}")
    
    return _bits_to_str(payload)