import librosa
import numpy as np
import msggan.spectrograms_helper as spec_helper


def output_audio(specs,_sample_rate=16000, n_fft=2048, hop_length_fft=256):
    spec = specs[0].data.cpu().numpy().T
    IF = specs[1].data.cpu().numpy().T
    back_mag, back_IF = spec_helper.melspecgrams_to_specgrams(spec, IF, _sample_rate, n_fft // 2)
    back_mag = np.vstack((back_mag, back_mag[-1]))
    back_IF = np.vstack((back_IF, back_IF[-1]))
    audio = mag_plus_phase(back_mag, back_IF, n_fft, hop_length_fft)
    return audio


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
    temp_mag = np.zeros(mag.shape, dtype=np.complex_)
    temp_phase = np.zeros(mag.shape, dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
            temp_mag[i,j] = np.complex(mag[i,j])

    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i, j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))

    return temp_mag * temp_phase


def mag_plus_phase(mag, IF, n_fft, hop_length_fft):

    mag = np.exp(mag) - 1.0e-6
    reconstruct_magnitude = np.abs(mag)

    reconstruct_phase_angle = np.cumsum(IF * np.pi, axis=1)
    stft = polar2rect(reconstruct_magnitude, reconstruct_phase_angle)
    inverse = librosa.istft(stft, hop_length=hop_length_fft, win_length=n_fft, window='hann')

    return inverse
