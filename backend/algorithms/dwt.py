import cv2, numpy as np, pywt

HEADER_BITS = 32

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

def embed_message(cover_image_path, message, output_path):
    img = cv2.imread(cover_image_path)
    if img is None:
        raise ValueError("Image not found.")
    YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCbCr[:,:,0].astype(np.float32)

    LL,(LH,HL,HH)=pywt.dwt2(Y,'haar')
    H, W = HL.shape
    HL = HL[:H-(H%2), :W-(W%2)]

    bits=_str_to_bits(message)
    header=_int_to_bits32(len(bits))
    payload=header+bits

    flat=HL.flatten().astype(np.int32)
    if len(payload)>flat.size:
        raise ValueError("Message too large.")
    flat[:len(payload)] = (flat[:len(payload)] & ~1) | np.array(payload, dtype=np.int32)
    HL2 = flat.reshape(HL.shape)

    Y2 = pywt.idwt2((LL,(LH,HL2,HH)),'haar')
    # fix broadcast mismatch by resizing to original
    Y2 = cv2.resize(Y2, (Y.shape[1], Y.shape[0]))
    YCbCr[:,:,0] = np.clip(Y2, 0, 255)
    stego = cv2.cvtColor(YCbCr, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, stego)

def extract_message(stego_path):
    img=cv2.imread(stego_path)
    if img is None: raise ValueError("Could not open stego image.")
    YCbCr=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    Y=YCbCr[:,:,0].astype(np.float32)

    LL,(LH,HL,HH)=pywt.dwt2(Y,'haar')
    flat=HL.flatten().astype(np.int32)
    header_bits=(flat[:HEADER_BITS]&1).tolist()
    msg_len=_bits32_to_int(header_bits)
    msg_bits=(flat[HEADER_BITS:HEADER_BITS+msg_len]&1).tolist()
    return _bits_to_str(msg_bits)
