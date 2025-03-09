import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Function to extract LPC coefficients and formants
def extract_formants(signal, sr, order=12):
    signal = signal * np.hamming(len(signal))
    a = librosa.lpc(y=signal, order=order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    formants = np.angle(roots) * (sr / (2 * np.pi))
    formants = np.sort(formants)
    return formants[:3] if len(formants) >= 3 else [0, 0, 0]

# Function to compute fundamental frequency F0
def extract_f0(signal, sr):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    peaks, _ = find_peaks(autocorr, height=0)
    if len(peaks) > 1:
        f0 = sr / peaks[0]
    else:
        f0 = 0
    return f0

# Rename each audio file with their label as the first character and save them all in a common folder and add the path to that folder below
dataset_path = "drive/MyDrive/Vowel_Dataset"
vowel_labels = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
features, labels = [], []

for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            label = file[0]
            if label in vowel_labels:
                signal, sr = librosa.load(os.path.join(subdir, file), sr=None)
                F1, F2, F3 = extract_formants(signal, sr)
                F0 = extract_f0(signal, sr)
                features.append([F1, F2, F3, F0])
                labels.append(vowel_labels[label])

features = np.array(features)
labels = np.array(labels)

# Visualization of extracted features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_names = ["F1", "F2", "F3", "F0"]

for i, ax in enumerate(axes.flatten()):
    for j, vowel in enumerate(vowel_labels.keys()):
        idx = np.where(labels == j)
        ax.scatter(idx, features[idx, i], label=vowel, alpha=0.5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(feature_names[i])
    ax.set_title(f"Distribution of {feature_names[i]} for Different Vowels")
    ax.legend()

plt.tight_layout()
plt.show()

# Histogram of fundamental frequencies
plt.figure(figsize=(8, 5))
plt.hist(features[:, 3], bins=30, alpha=0.7, color='b')
plt.xlabel("Fundamental Frequency (F0)")
plt.ylabel("Count")
plt.title("Distribution of Fundamental Frequencies")
plt.show()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=45)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=vowel_labels.keys(), zero_division=1))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vowel_labels.keys(), yticklabels=vowel_labels.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Vowel space visualization
plt.figure(figsize=(10, 6))
for i, vowel in enumerate(vowel_labels.keys()):
    idx = np.where(labels == i)
    plt.scatter(features[idx, 0], features[idx, 1], label=vowel, alpha=0.5)
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.title("Vowel Space (F1 vs F2)")
plt.show()