import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Compute amplitude
    amplitude = np.abs(y)

    # Compute pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])
    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]

    # Compute RMS energy
    rms_energy = librosa.feature.rms(y=y)[0]

    # Compute Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_db = librosa.power_to_db(mfccs)

    # Compute Fundamental Frequency (f0)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300)

    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot Amplitude over Time
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), amplitude, label='Amplitude', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # Enhanced Spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Log Scale)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    # Smoothed Zero-Crossing Rate
    zcr_smoothed = scipy.signal.savgol_filter(zcr, 11, 3)
    plt.figure(figsize=(10, 4))
    plt.plot(zcr_smoothed, label='Smoothed ZCR', color='green')
    plt.xlabel('Frames')
    plt.ylabel('Rate')
    plt.title('Zero-Crossing Rate Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # MFCCs Plot
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfccs_db, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar(label='MFCC (dB)')
    plt.title('MFCCs')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.show()

    # Plot Fundamental Frequency (f0)
    plt.figure(figsize=(10, 4))
    plt.plot(f0, label='Fundamental Frequency (f0)', color='red')
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    plt.title('Fundamental Frequency Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    return {
        'Amplitude': amplitude,
        'Pitch': pitch,
        'RMS Energy': rms_energy,
        'Zero-Crossing Rate': zcr,
        'MFCCs': mfccs,
        'Fundamental Frequency (f0)': f0,
        'Spectrogram': D
    }

audio_folder = "drive/MyDrive/Voice_Dataset"
for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        file_path = os.path.join(audio_folder, file)
        print(f"Processing: {file}")
        results = process_audio(file_path)
        print(f"Pitch Mean: {np.mean(results['Pitch']) if len(results['Pitch']) > 0 else 'N/A'} Hz")
        print(f"RMS Energy Mean: {np.mean(results['RMS Energy'])}")
        print(f"Zero-Crossing Rate Mean: {np.mean(results['Zero-Crossing Rate'])}")
        print(f"MFCCs Shape: {results['MFCCs'].shape}")
        print(f"Mean Fundamental Frequency (f0): {np.nanmean(results['Fundamental Frequency (f0)'])} Hz")
        print("\n")

"""The 10 different audio files are:
1. News Report (Neutral, Moderate Volume, Medium Pitch)
2. Excited Announcement (High Pitch, Loud Volume)
3. Whispered Secret (Low Pitch, Soft Volume)
4. Formal Speech (Medium Pitch, Moderate Volume, Slow Pace)
5. Angry Complaint (Low Pitch, Loud Volume, Fast Pace)
6. Childlike Excitement (High Pitch, Soft Volume, Fast Pace)
7. Robotic/Monotone (Flat Pitch, Moderate Volume, Even Pace)
8. Dramatic Storytelling (Varied Pitch and Volume, Expressive Tone)
9. Relaxing Meditation Guide (Low Pitch, Soft Volume, Slow Pace)
10. Sarcastic Remark (Medium Pitch, Moderate Volume, Slow Drawl)

The file names give a description of what the voice notes are about. We can check if the same are followed in the plots and results.

The audio characteristics shown here for all my 10 different audio files are:
1. Amplitude
2. Pitch
3. RMS Energy
4. Zero-Crossing Rate (ZCR)
5. Mel-Frequency Cepstral Coefficients (MFCCs)
6. Fundamental Frequency (f0)
7. Spectrogram
"""