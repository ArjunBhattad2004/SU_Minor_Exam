import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import noisereduce as nr

dataset_path = "drive/MyDrive/Speeches of leaders"

audio_files = [f for f in os.listdir(dataset_path) if f.endswith(('.wav', '.mp3'))]
print("Available audio files:", audio_files)

def extract_features(audio_path, max_length=300):
    y, sr = librosa.load(audio_path, sr=None)

    # Apply noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_denoised = librosa.util.normalize(y_denoised)

    # Ensure fixed duration (truncate or pad)
    max_samples = sr * max_length
    if len(y_denoised) > max_samples:
        y_denoised = y_denoised[:max_samples]
    else:
        y_denoised = np.pad(y_denoised, (0, max_samples - len(y_denoised)))

    # Compute Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y_denoised)[0]

    # Compute Short-Time Energy (STE)
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        sum(abs(y_denoised[i:i+frame_length]**2))
        for i in range(0, len(y_denoised), hop_length)
    ])

    # Compute MFCCs (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)

    return zcr, energy, mfccs, sr, y_denoised

def plot_features(zcr, energy, mfccs, sr, y, title="Audio Features"):
    plt.figure(figsize=(14, 8))

    # Waveform Plot
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.75, color="royalblue")
    plt.title(f"Waveform - {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Zero-Crossing Rate Plot
    plt.subplot(3, 1, 2)
    zcr_smoothed = np.convolve(zcr, np.ones(10)/10, mode="same")
    plt.plot(zcr_smoothed, label="ZCR (Smoothed)", color="darkorange", lw=1.5)
    plt.title("Zero-Crossing Rate (ZCR)", fontsize=14, fontweight='bold')
    plt.xlabel("Frames")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Short-Time Energy(STE)
    plt.subplot(3, 1, 3)
    energy /= np.max(energy)
    plt.fill_between(range(len(energy)), energy, color="red", alpha=0.4, label="STE")
    plt.plot(energy, color="darkred", lw=1.5)
    plt.title("Short-Time Energy (STE)", fontsize=14, fontweight='bold')
    plt.xlabel("Frames")
    plt.ylabel("Energy (Normalized)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # MFCC Spectrogram
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mfccs, x_axis='time', cmap='magma', sr=sr)
    plt.colorbar(label="Coefficient Magnitude")
    plt.title("Mel-Frequency Cepstral Coefficients (MFCCs)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# Choosing an audio file from the dataset
audio_file = os.path.join(dataset_path, audio_files[0])

print(f"Processing {audio_file}")
zcr, energy, mfccs, sr, y = extract_features(audio_file)

plot_features(zcr, energy, mfccs, sr, y, title=audio_files[0])

calm_speech_file = "drive/MyDrive/Speeches of leaders/Mahatma Gandhi.mp3"
energetic_speech_file = "DO IT FOR YOU - Motivational Speech.mp3"

zcr_calm, energy_calm, mfccs_calm, sr_calm, y_calm = extract_features(calm_speech_file)
zcr_energy, energy_energy, mfccs_energy, sr_energy, y_energy = extract_features(energetic_speech_file)

def compare_features(zcr_calm, zcr_energy, energy_calm, energy_energy, mfccs_calm, mfccs_energy):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Setting Y-axis limits to be the same for better comparison
    zcr_max = max(np.max(zcr_calm), np.max(zcr_energy))
    energy_max = max(np.max(energy_calm), np.max(energy_energy))

    # Zero-Crossing Rate (ZCR)
    axes[0].plot(zcr_calm, label="Calm Speech", color='blue')
    axes[0].set_title("Zero-Crossing Rate (Calm)")
    axes[0].legend()
    axes[0].set_ylim(0, zcr_max)

    axes[1].plot(zcr_energy, label="Energetic Speech", color='red')
    axes[1].set_title("Zero-Crossing Rate (Energetic)")
    axes[1].legend()
    axes[1].set_ylim(0, zcr_max)

    # Short-Time Energy (STE)
    axes[2].plot(energy_calm, label="Calm Speech", color='blue')
    axes[2].set_title("Short-Time Energy (Calm)")
    axes[2].legend()
    axes[2].set_ylim(0, energy_max)

    axes[3].plot(energy_energy, label="Energetic Speech", color='red')
    axes[3].set_title("Short-Time Energy (Energetic)")
    axes[3].legend()
    axes[3].set_ylim(0, energy_max)

    # MFCCs - Display as Spectrogram
    img1 = librosa.display.specshow(mfccs_calm, ax=axes[4], x_axis='time', cmap='viridis')
    axes[4].set_title("MFCCs (Calm Speech)")
    fig.colorbar(img1, ax=axes[4])

    img2 = librosa.display.specshow(mfccs_energy, ax=axes[5], x_axis='time', cmap='magma')
    axes[5].set_title("MFCCs (Energetic Speech)")
    fig.colorbar(img2, ax=axes[5])

    plt.tight_layout()
    plt.show()

compare_features(zcr_calm, zcr_energy, energy_calm, energy_energy, mfccs_calm, mfccs_energy)