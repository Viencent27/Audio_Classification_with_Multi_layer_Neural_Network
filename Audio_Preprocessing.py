import librosa
import soundfile as sf
import numpy as np

def preprocess_audio(audio_path, target_sample_rate):
	"""
	Melakukan preprocessing audio dengan menghitung fitur MFCC dari file audio.
	Mengubah sampling rate audio menjadi target_sample_rate.
	"""
	signal, sample_rate = sf.read(audio_path)
	signal_resampled = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
	signal_normalized = librosa.util.normalize(signal_resampled)
	mfccs = librosa.feature.mfcc(y=signal_normalized, sr=target_sample_rate, n_mfcc=13)
	mfccs_mean = np.mean(mfccs, axis=1)
	return mfccs_mean, mfccs