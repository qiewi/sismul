import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

# Parameter
fs = 48000  # sample rate
duration = 2  # seconds
f = 440  # frequency (Hz)
seed = 8 * 100 + 25  # bulan=8, tanggal=25 → 825

# Generate sinyal asli
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
signal = np.sin(2 * np.pi * f * t)

# Generate watermark (spread spectrum noise)
np.random.seed(seed)
watermark = np.random.choice([-1, 1], size=signal.shape)

# Embedding watermark
alpha1 = 0.01
alpha2 = 0.1
signal_wm1 = signal + alpha1 * watermark
signal_wm2 = signal + alpha2 * watermark

# Plot
plt.figure(figsize=(10, 7))
plt.subplot(3,1,1)
plt.plot(t[:1000], signal[:1000])
plt.title('Sinyal Asli (440 Hz)')
plt.xlabel('Waktu (sampel)')
plt.ylabel('Amplitudo')

plt.subplot(3,1,2)
plt.plot(t[:1000], signal_wm1[:1000], color='orange')
plt.title(f'Sinyal dengan Watermark (α = {alpha1})')
plt.xlabel('Waktu (sampel)')
plt.ylabel('Amplitudo')

plt.subplot(3,1,3)
plt.plot(t[:1000], signal_wm2[:1000], color='green')
plt.title(f'Sinyal dengan Watermark (α = {alpha2})')
plt.xlabel('Waktu (sampel)')
plt.ylabel('Amplitudo')

plt.tight_layout()
plt.savefig('watermarking_plot.png')

# Simpan dan mainkan suara
write('original.wav', fs, (signal * 32767).astype(np.int16))
write('watermarked_001.wav', fs, (signal_wm1 * 32767).astype(np.int16))
write('watermarked_01.wav', fs, (signal_wm2 * 32767).astype(np.int16))

print("Mainkan suara asli...")
sd.play(signal, fs)
sd.wait()
print("Mainkan suara dengan watermark α=0.01...")
sd.play(signal_wm1, fs)
sd.wait()
print("Mainkan suara dengan watermark α=0.1...")
sd.play(signal_wm2, fs)
sd.wait()

# Deteksi watermark (korelasi)
corr1 = np.dot(signal_wm1, watermark)
corr2 = np.dot(signal_wm2, watermark)
print(f"Hasil deteksi watermark (α={alpha1}): {corr1:.2f}")
print(f"Hasil deteksi watermark (α={alpha2}): {corr2:.2f}")
