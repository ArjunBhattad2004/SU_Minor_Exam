# **Speech Understanding: Midsem Exam**

1. Collect a dataset of 10 audio samples of your own voice in which you speak different kinds of text at
different pitch and volume. Process them to perform the following tasks:
a. Compute the amplitude, pitch, frequency, and RMS energy. Analyze different characteristics of
the audio signal and draw the spectrogram. (10 marks)
2. Imagine you are tasked with analyzing a collection of audio recordings from a historical archive (you
have to use the dataset available in the mentioned drive link: Speeches of leaders ). These
recordings feature speeches from prominent figures of the early 20th century, and you are
specifically interested in understanding the emotional tone and speaking style of these individuals.
Due to the limitations of the era, the audio quality is often poor, with background noise and
variations in recording volume. Perform the following tasks: (40 marks)
1. Audio Feature Extraction:
o Write a Python script that can read a given audio file (e.g., in .wav, .mp3 format) and
extract the following traditional speech features:
▪ Zero-Crossing Rate (ZCR)
▪ Short-Time Energy (STE)
▪ Mel-Frequency Cepstral Coefficients (MFCCs) (first 13 coefficients)
o Your script should handle potential issues like varying audio lengths and noise.
2. Comparative Analysis:
o Select two distinct audio clips from a provided sample dataset (or find two short clips
online). One clip should feature a speech with a perceived "calm and formal" tone,
and the other with a "passionate and energetic" tone.
o Using your Python script, extract the features for both clips.
o Analyze and compare the extracted ZCR, STE, and MFCC features between the two
clips. Explain how the differences in these features correlate with the perceived
emotional tones.
o Specifically, discuss how the following characteristics of the two tones would be
represented in the extracted features:
▪ The energy of the signal.
▪ The frequency content of the signal.
▪ The general shape of the vocal tract.

3. Limitations and Improvements:
o Discuss the limitations of using traditional speech features for emotional tone
analysis, especially in the context of historical recordings with poor audio quality.
o Suggest one potential improvement or alternative approach that could enhance the
analysis, considering the challenges posed by the historical nature of the recordings.

Submission:
● Submit your Python script.

● Include a report explaining your analysis, feature comparisons, and discussion of limitations
and improvements.
● Include graphs of the extracted features.
3. Develop a Python-based system that can classify five different vowel sounds (/a/, /e/, /i/, /o/, /u/)
using traditional speech processing techniques. Your system should:
1. Extract formant frequencies (F1, F2, F3) and fundamental frequency (F0) from provided
speech samples
2. Visualize the vowel space using the first two formants (F1-F2 plot)
3. Implement a classification algorithm using these acoustic features
4. Evaluate the performance using confusion matrices and accuracy metrics. (50 marks)
You have to download and use this dataset (Vowel database: adults) for this question. Use all the
categories for this dataset (Adult males, Adult Females), and divide the dataset into train and test
split only in a ratio of 80:20 with a random_state of 45.
Requirements: Your solution should include the following components:
Part 1: Feature Extraction (40%)
● Write code to load audio files and perform pre-processing (framing, windowing)
● Implement Linear Predictive Coding (LPC) to extract formant frequencies
● Extract fundamental frequency using autocorrelation or another appropriate method
● Create visualizations showing the extracted features for different vowels
Part 2: Classification System (40%)
● Implement a classification algorithm using the extracted features (consider K-nearest
neighbors, Gaussian Mixture Models, or a simple threshold-based approach)
● Train your classifier on the provided training data
● Test the classifier on the separate test set
● Generate and analyze a confusion matrix
Part 3: Analysis and Reflection (20%)
● Compare your results with theoretical expectations for vowel formant patterns
● Discuss potential sources of error or confusion in your classification system
● Explain how your approach relates to historical speech recognition systems
● Suggest at least two improvements that could enhance classification accuracy
Deliverables
1. Complete Python code with comments explaining your approach
2. Visualizations of formant spaces and feature distributions
3. Classification results and performance metrics
4. A brief report (2-3 pages) discussing your methodology and findings
