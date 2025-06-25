import os
from pydub import AudioSegment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.io import wavfile

# Paths
wav_path = '../public/ads.wav'
mp3_path = '../public/ads.mp3'

# 1. Kompresi WAV ke MP3
audio = AudioSegment.from_wav(wav_path)
audio.export(mp3_path, format='mp3', bitrate='192k')

# 2. Bandingkan ukuran file
wav_size = os.path.getsize(wav_path)
mp3_size = os.path.getsize(mp3_path)
ratio = 100 * (1 - mp3_size / wav_size)
print(f"Ukuran WAV: {wav_size/1024:.2f} KB")
print(f"Ukuran MP3: {mp3_size/1024:.2f} KB")
print(f"Rasio Kompresi: {ratio:.2f}%")

# 3. Analisis dan visualisasi
# Fungsi untuk mendapatkan data waveform dan sample rate dari file audio
def get_waveform_and_sr(path):
    if path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(path)
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)  # convert to mono
        return samples, sr
    else:
        sr, samples = wavfile.read(path)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return samples, sr

wav_samples, wav_sr = get_waveform_and_sr(wav_path)
mp3_samples, mp3_sr = get_waveform_and_sr(mp3_path)

# Plot waveform
def plot_waveform(samples, sr, title, subplot):
    t = np.arange(len(samples)) / sr
    plt.subplot(2, 1, subplot)
    plt.plot(t, samples)
    plt.title(title)
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Amplitudo')

plt.figure(figsize=(12, 6))
plot_waveform(wav_samples, wav_sr, 'Gelombang WAV Asli', 1)
plot_waveform(mp3_samples, mp3_sr, 'Gelombang MP3 Kompresi', 2)
plt.tight_layout()
plt.savefig('waveform.png')

# Plot frequency spectrum
def plot_spectrum(samples, sr, title, subplot):
    N = len(samples)
    yf = np.abs(fft(samples))[:N//2]
    xf = fftfreq(N, 1/sr)[:N//2]
    plt.subplot(2, 1, subplot)
    plt.plot(xf, yf)
    plt.title(title)
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Magnitudo')
    plt.xlim(0, sr/2)

plt.figure(figsize=(12, 6))
plot_spectrum(wav_samples, wav_sr, 'Spektrum Frekuensi WAV Asli', 1)
plot_spectrum(mp3_samples, mp3_sr, 'Spektrum Frekuensi MP3 Kompresi', 2)
plt.tight_layout()
plt.savefig('spectrum.png')
