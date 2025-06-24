import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

# Step ke-1: Memuat gambar dari public/selfie.jpeg
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    y_component = image_yuv[:, :, 0].astype(np.float32)
    
    print(f"Ukuran gambar: {image_rgb.shape}")
    print(f"Komponen Y: {y_component.shape}")
    return y_component

# Step ke-2: Membuat gambar menjadi makroblok 8x8
def create_macroblocks(y_component):
    height, width = y_component.shape
    
    # Padding agar bisa dibagi 8
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8
    padded_y = np.pad(y_component, ((0, pad_height), (0, pad_width)), mode='edge')
    new_height, new_width = padded_y.shape
    
    # Bagi menjadi blok 8x8
    blocks = []
    for i in range(0, new_height, 8):
        for j in range(0, new_width, 8):
            block = padded_y[i:i+8, j:j+8]
            blocks.append(block)
    
    blocks_array = np.array(blocks)
    print(f"Jumlah blok 8x8: {len(blocks)}")
    print(f"Blok pertama:\n{blocks_array[0].astype(int)}")
    return blocks_array, (new_height, new_width)

# Step ke-3: Shift nilai dengan -128
def shift_minus_128(blocks):
    shifted_blocks = blocks - 128
    print(f"Setelah shift -128:\n{shifted_blocks[0].astype(int)}")
    return shifted_blocks

# Step ke-4: Menerapkan DCT
def apply_dct(blocks):
    dct_blocks = np.zeros_like(blocks)
    for i, block in enumerate(blocks):
        dct_blocks[i] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    print(f"Setelah DCT:\n{dct_blocks[0].astype(int)}")
    return dct_blocks

# Step ke-5: Kuantisasi
def quantize(dct_blocks):
    # Matriks kuantisasi JPEG standar
    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    quantized_blocks = np.zeros_like(dct_blocks)
    for i, block in enumerate(dct_blocks):
        quantized_blocks[i] = np.round(block / q_matrix)
    
    print(f"Setelah kuantisasi:\n{quantized_blocks[0].astype(int)}")
    return quantized_blocks, q_matrix

# Ekstrak komponen DC dan AC
def extract_dc_ac(quantized_blocks):
    dc_components = []
    ac_components = []
    
    # Pola zigzag untuk AC
    zigzag = [
        (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1),
        (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4),
        (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2), (3,3),
        (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4), (4,3),
        (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4), (3,5),
        (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2),
        (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6), (6,5),
        (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    
    for block in quantized_blocks:
        # DC adalah elemen [0,0]
        dc_components.append(int(block[0, 0]))
        
        # AC mengikuti pola zigzag
        ac_block = []
        for row, col in zigzag:
            ac_block.append(int(block[row, col]))
        ac_components.extend(ac_block)
    
    return dc_components, ac_components

# Encoding DC dengan DPCM
def encode_dc(dc_components):
    dpcm_values = []
    prev_dc = 0
    for dc in dc_components:
        diff = dc - prev_dc
        dpcm_values.append(diff)
        prev_dc = dc
    return dpcm_values

# Encoding AC dengan Run-Length
def encode_ac(ac_components):
    encoded = []
    i = 0
    while i < len(ac_components):
        if ac_components[i] == 0:
            zero_count = 0
            while i < len(ac_components) and ac_components[i] == 0:
                zero_count += 1
                i += 1
            
            while zero_count > 15:
                encoded.extend([15, 0])
                zero_count -= 15
            
            if zero_count > 0 and i < len(ac_components):
                encoded.extend([zero_count, ac_components[i]])
                i += 1
            elif zero_count > 0:
                encoded.extend([zero_count, 0])
        else:
            encoded.extend([0, ac_components[i]])
            i += 1
    return encoded

# Konversi ke binary
def to_binary(values, bits=8):
    binary_string = ""
    for value in values:
        if value < 0:
            value = (1 << bits) + value
        binary = format(value & ((1 << bits) - 1), f'0{bits}b')
        binary_string += binary
    return binary_string

# Tampilkan output binary
def show_binary_output(dc_binary, ac_binary):
    print("\n" + "="*80)
    print(f"{'OUTPUT':^80}")
    print("="*80)
    print(f"{'DC BINARY':^25} | {'AC BINARY':^50}")
    print("-"*80)
    
    dc_display = dc_binary[:10] if len(dc_binary) > 10 else dc_binary
    ac_display = ac_binary[:60] if len(ac_binary) > 60 else ac_binary
    
    print(f"{dc_display:<25} | {ac_display:<50}")
    print("="*80)

# Step ke-6: De-kuantisasi
def dequantize(quantized_blocks, q_matrix):
    dequantized_blocks = np.zeros_like(quantized_blocks)
    for i, block in enumerate(quantized_blocks):
        dequantized_blocks[i] = block * q_matrix
    
    print(f"Setelah de-kuantisasi:\n{dequantized_blocks[0].astype(int)}")
    return dequantized_blocks

# Step ke-7: Menerapkan IDCT
def apply_idct(dct_blocks):
    idct_blocks = np.zeros_like(dct_blocks)
    for i, block in enumerate(dct_blocks):
        idct_blocks[i] = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    print(f"Setelah IDCT:\n{idct_blocks[0].astype(int)}")
    return idct_blocks

# Step ke-8: Shift kembali dengan +128
def shift_plus_128(blocks):
    shifted_blocks = blocks + 128
    shifted_blocks = np.clip(shifted_blocks, 0, 255)
    print(f"Setelah shift +128:\n{shifted_blocks[0].astype(int)}")
    return shifted_blocks

# Rekonstruksi gambar dari blok
def reconstruct_image(blocks, image_shape):
    height, width = image_shape
    reconstructed = np.zeros((height, width))
    
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            reconstructed[i:i+8, j:j+8] = blocks[block_idx]
            block_idx += 1
    
    return reconstructed.astype(np.uint8)

def main():
    print("=== Kompresi Gambar dengan Makroblok ===\n")
    
    # Step 1: Memuat gambar
    print("Step 1: Memuat gambar...")
    y_component = load_image('../public/selfie.jpeg')
    
    # Step 2: Membuat makroblok 8x8
    print("\nStep 2: Membuat makroblok 8x8...")
    blocks, image_shape = create_macroblocks(y_component)
    
    # Step 3: Shift -128
    print("\nStep 3: Shift nilai dengan -128...")
    shifted_blocks = shift_minus_128(blocks)
    
    # Step 4: DCT
    print("\nStep 4: Menerapkan DCT...")
    dct_blocks = apply_dct(shifted_blocks)
    
    # Step 5: Kuantisasi
    print("\nStep 5: Kuantisasi...")
    quantized_blocks, q_matrix = quantize(dct_blocks)
    
    # Ekstrak dan encode komponen DC/AC
    print("\nEkstrak komponen DC dan AC...")
    dc_components, ac_components = extract_dc_ac(quantized_blocks)
    
    print("Encoding komponen DC dengan DPCM...")
    dpcm_dc = encode_dc(dc_components)
    
    print("Encoding komponen AC dengan RLE...")
    rle_ac = encode_ac(ac_components)
    
    print("Konversi ke binary...")
    dc_binary = to_binary(dpcm_dc)
    ac_binary = to_binary(rle_ac)
    
    # Tampilkan output binary
    show_binary_output(dc_binary, ac_binary)
    
    # Step 6: De-kuantisasi
    print("\nStep 6: De-kuantisasi...")
    dequantized_blocks = dequantize(quantized_blocks, q_matrix)
    
    # Step 7: IDCT
    print("\nStep 7: Menerapkan IDCT...")
    idct_blocks = apply_idct(dequantized_blocks)
    
    # Step 8: Shift +128
    print("\nStep 8: Shift kembali dengan +128...")
    final_blocks = shift_plus_128(idct_blocks)
    
    # Rekonstruksi gambar
    print("\nRekonstruksi gambar...")
    reconstructed_image = reconstruct_image(final_blocks, image_shape)
    
    # Simpan hasil
    original_reconstructed = reconstruct_image(blocks, image_shape)
    cv2.imwrite('original_y.png', original_reconstructed)
    cv2.imwrite('reconstructed_y.png', reconstructed_image)
    
    # Hitung MSE
    mse = np.mean((original_reconstructed.astype(float) - reconstructed_image.astype(float)) ** 2)
    print(f"\nMean Squared Error: {mse:.2f}")
    print("Gambar disimpan: original_y.png, reconstructed_y.png")
    
    return {
        'step1_original': y_component.astype(int),
        'step2_blocks': blocks.astype(int),
        'step3_shifted': shifted_blocks.astype(int),
        'step4_dct': dct_blocks.astype(int),
        'step5_quantized': quantized_blocks.astype(int),
        'step6_dequantized': dequantized_blocks.astype(int),
        'step7_idct': idct_blocks.astype(int),
        'step8_final': final_blocks.astype(int),
        'dc_binary': dc_binary,
        'ac_binary': ac_binary
    }

if __name__ == "__main__":
    results = main()
