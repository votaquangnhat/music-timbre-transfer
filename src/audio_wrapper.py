import numpy as np
from PIL import Image
import torch
import torchaudio
import torchaudio.transforms as T
from IPython.display import Audio, display
import librosa

print("Audio processing utilities loaded.")

# in this file
# spectrogram is (channel, frequency, time)
# PIL image after turn to np is (height, width, channel)

class SpectrogramConverter: # helper class

    def __init__(
            self, 
            image_width=512, # x_res
            n_mels=512, # y_res
            sample_rate=44100,
            n_fft=17640,
            hop_length=441,
            win_length=4410, # config for riffusion
    ):

        self.image_width = image_width
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def spectrogram_from_image(
            self,
            image: Image.Image, 
            max_volume: float = 50, 
            power_for_image: float = 0.25
    ) -> np.ndarray:

        data = np.array(image).astype(np.float32)
        data = data[::-1, :, 0]
        data = 255 - data
        data = data * max_volume / 255
        data = np.power(data, 1 / power_for_image)

        print(f"Spectrogram shape from image: {data.shape}")

        return data
    
    def image_from_spectrogram(
            self,
            spectrogram: np.ndarray,
            max_volume: float = 50,
            power_for_image: float = 0.25
        ) -> Image.Image:
        """return pil rgb image"""
        data = np.power(spectrogram, power_for_image)
        data = data * 255 / max_volume
        data = 255 - data
        image = Image.fromarray(data.astype(np.uint8))
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image = image.convert("RGB")

        return image

    def spectrogram_from_waveform(
            self, 
            waveform: np.ndarray,
            power = None,
            mel_scale: bool = True
    ) -> np.ndarray:

        spectrogram_func = T.Spectrogram(
            n_fft=self.n_fft,
            power=power,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
        Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

        Sxx_mag = np.abs(Sxx_complex)

        if mel_scale:
            mel_scaler = T.MelScale(
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=self.n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
            )

            Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

        return Sxx_mag
    
    def waveform_from_spectrogram(
            self,
            Sxx: np.ndarray,
            mel_scale: bool = True,
            num_griffin_lim_iters: int = 32,
            device: str = "cuda",
    ) -> np.ndarray:

        Sxx_torch = torch.from_numpy(Sxx).to(device)

        if mel_scale:
            mel_inv_scaler = T.InverseMelScale(
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=self.n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
            ).to(device)

            Sxx_torch = mel_inv_scaler(Sxx_torch)

        griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1.0,
            n_iter=num_griffin_lim_iters,
        ).to(device)

        waveform = griffin_lim(Sxx_torch).cpu().numpy()

        return waveform
    
    def waveform_from_image(self, image: Image.Image) -> np.ndarray:
        return self.waveform_from_spectrogram(self.spectrogram_from_image(image))


class AudioWrapper:

    def __init__(self):
        self.converter = SpectrogramConverter()
        self.audio = None
        # this calculation of slice size will make sure created image has x_res = image_width
        self.slice_size = (self.converter.image_width) * self.converter.hop_length - 1
        self.number_of_slices = 0
        self.sample_rate = self.converter.sample_rate

    def load_audio(self, audio_path: str) -> np.ndarray:
        self.audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        self.number_of_slices = len(self.audio) // self.slice_size
        return self.audio
    
    def get_audio_slices(self, index: int) -> np.ndarray:
        if self.audio is None:
            raise ValueError("Audio not loaded. Please call load_audio() first.")
        if self.number_of_slices == 0:
            return self.audio # too short
        if index < 0 or index >= self.number_of_slices:
            raise IndexError(f"Index out of range. Must be between 0 and {self.number_of_slices - 1}.")
        
        start = index * self.slice_size
        end = start + self.slice_size
        return self.audio[start:end]
    
    def display_audio(self, audio):
        audio = Audio(audio, rate=self.sample_rate)
        display(audio)
    
    def spectrogram_from_audio_slice(self, index: int) -> np.ndarray:
        """
        Guarantees the output spectrogram has shape (n_mels, image_width)
        """
        audio_slice = self.get_audio_slices(index)
        result = self.converter.spectrogram_from_waveform(audio_slice)
        if result.shape != (self.converter.n_mels, self.converter.image_width):
            print(f'Length of audio slice: {len(audio_slice)}')
            raise ValueError(f"Unexpected spectrogram shape: {result.shape}. Expected ({self.converter.n_mels}, {self.converter.image_width}).")
        return result

    def save_spectrogram(self, spectrogram: np.ndarray, path: str):
        np.save(path, spectrogram)