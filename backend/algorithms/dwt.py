import cv2
import numpy as np
import pywt

HEADER_BITS = 32
STRENGTH = 8  # Embedding strength for DWT

def _str_to_bits(s): 
    return [int(x) for x in ''.join(format(ord(c),'08b') for c in s)]

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8: break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def _int_to_bits32(n): 
    return [int(x) for x in format(n,'032b')]

def _bits32_to_int(bits): 
    return int(''.join(map(str,bits[:32])),2)

def embed_message(cover_image_path, message, output_path, strength=STRENGTH):
    img = cv2.imread(cover_image_path)
    if img is None:
        raise ValueError("Image not found.")
    
    YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCbCr[:,:,0].astype(np.float32)
    
    # Store original shape for reconstruction
    orig_shape = Y.shape

    # Perform DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')
    
    # Prepare message
    bits = _str_to_bits(message)
    header = _int_to_bits32(len(bits))
    payload = header + bits

    # Embed in HL band using quantization
    flat = HL.flatten()
    if len(payload) > flat.size:
        raise ValueError(f"Message too large. Capacity={flat.size} bits, need={len(payload)}")
    
    # Stronger embedding using quantization
    for i in range(len(payload)):
        coeff = flat[i]
        target_bit = payload[i]
        
        # Quantize to multiple of strength*2, encode bit in remainder
        quantized = round(coeff / (strength * 2)) * (strength * 2)
        new_coeff = quantized + (target_bit * strength) + (strength // 2)
        flat[i] = new_coeff
    
    HL_modified = flat.reshape(HL.shape)

    # Inverse DWT
    Y2 = pywt.idwt2((LL, (LH, HL_modified, HH)), 'haar')
    
    # Handle size mismatch from DWT/IDWT
    if Y2.shape != orig_shape:
        Y2 = Y2[:orig_shape[0], :orig_shape[1]]
    
    # Update image
    YCbCr[:,:,0] = np.clip(Y2, 0, 255).astype(np.uint8)
    stego = cv2.cvtColor(YCbCr, cv2.COLOR_YCrCb2BGR)
    
    # Save with no compression
    cv2.imwrite(output_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Embedded {len(payload)} bits ({len(message)} chars) with strength={strength}")

def extract_message(stego_path, strength=STRENGTH):
    img = cv2.imread(stego_path)
    if img is None: 
        raise ValueError("Could not open stego image.")
    
    YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCbCr[:,:,0].astype(np.float32)

    # Perform DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, 'haar')
    
    # Extract bits using quantization
    flat = HL.flatten()
    bits = []
    
    for i in range(len(flat)):
        coeff = flat[i]
        # Extract bit based on quantization remainder
        remainder = round(coeff) % (strength * 2)
        bit = 1 if remainder >= strength else 0
        bits.append(bit)
    
    # Read header
    header_bits = bits[:HEADER_BITS]
    msg_len = _bits32_to_int(header_bits)
    
    print(f"Header says message length: {msg_len} bits")
    
    if HEADER_BITS + msg_len > len(bits):
        raise ValueError(f"Message length {msg_len} exceeds capacity.")
    
    # Extract message bits
    msg_bits = bits[HEADER_BITS:HEADER_BITS + msg_len]
    
    print(f"Extracted {len(msg_bits)} bits")
    print(f"First 64 bits: {msg_bits[:64]}")
    
    return _bits_to_str(msg_bits)