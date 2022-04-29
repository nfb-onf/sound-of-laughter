import numpy as np
import librosa
import torch
from pyrubberband import pyrb

class AddNoise:

    def __init__(self, strength=0.01):
        self.strength = strength
        
    def __call__(self, wav_data, sr=None):
        return wav_data + np.random.randn(*wav_data.shape) * self.strength


class Shifting:

    def __init__(self, shift_max=0.2, shift_direction='random'):
        """
        random shift in direction 'shift_direction' up to shift_max seconds 
        """
        self.shift_max = shift_max
        self.shift_direction = shift_direction

    def __call__(self, wav_data, sr):
        shift = np.random.randint(1, int(sr * self.shift_max))

        if self.shift_direction == 'random':
            if np.random.randint(0, 2):
                shift_direction = 'right'
            else:
                shift_direction = 'left'
        else:
            shift_direction = self.shift_direction

        pad = np.mean(wav_data)
        out = np.zeros(wav_data.shape)
        if shift_direction == 'right':
            out[shift:] = wav_data[:-shift]
            out[:shift] = pad
        else:
            out[:-shift] = wav_data[shift:]
            out[-shift:] = pad

        return out


class ChangePitch:

    def __init__(self, max_pitch=3):
        self.max_pitch = max_pitch
 
    def __call__(self, wav_data, sr):
        pitch_increase = np.random.uniform(-1,1) * self.max_pitch
        out = pyrb.pitch_shift(wav_data, sr, pitch_increase)
        return out


class ChangeSpeed:

    def __init__(self, max_speed_change=0.2, direction='random'):
        self.max_speed = max_speed_change
        self.direction = direction

    def __call__(self, wav_data, sr):

        if self.direction == 'random':
            speed = float(np.random.uniform(-1.0,1.0) * self.max_speed)
            
        elif self.direction == 'down':
            speed = float(np.random.uniform(-1.0,0.0) * self.max_speed)
        elif self.direction == 'up':
            speed = float(np.random.uniform(0.0,1.0) * self.max_speed)
        else:
            raise ValueError('unknown direction "{}"'.format(self.direction))

        out = pyrb.time_stretch(wav_data, sr, 1.0 + speed)

        return out


class Compose:

    def __init__(self, transforms):
        # transforms is list of transform functions
        self.transforms = transforms

    def __call__(self, wav_data, sr):
        out = wav_data
        for t in self.transforms: 
            out = t(out, sr)
            
        return out


class RandomTransform:

    def __init__(self, transforms):

        self.transforms = transforms
        
    def __call__(self, wav_data, sr):

        n_transforms = np.random.randint(0, len(self.transforms))
        list_transforms = np.random.permutation(range(len(self.transforms)))[:n_transforms]

        out = wav_data
        for idx in list_transforms:
            t = self.transforms[idx]
            out = t(out, sr)
        return out


class ToTensor:

    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad
        pass

    def __call__(self, wav_data, sr=None):
        out = torch.tensor(wav_data, requires_grad=self.requires_grad).unsqueeze(0)
        return out
