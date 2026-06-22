import gin
gin.enter_interactive_mode()

from IPython.display import display, Audio
import torch
import numpy as np
import librosa
from diffusion.model import EDM_ADV
import sys


torch.set_grad_enabled(False)

# Import paths
folder = "./pretrained/slakh/"
checkpoint_path = folder + "checkpoint.pt"
config = folder + "config.gin"

autoencoder_path = "./pretrained/AE_slakh.pt"

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

from diffusion.model import EDM_ADV

# Parse config
gin.parse_config_file(config)
SR = gin.query_parameter("%SR")
audio_length = gin.query_parameter("%X_LENGTH")

# Instantiate model
blender = EDM_ADV()

# Load checkpoints
state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
blender.load_state_dict(state_dict, strict=False)

emb_model = torch.jit.load(autoencoder_path).eval().to(device)

# Send to device
blender = blender.eval().to(device)

def load_audio(path, sr):
    audio_full, sr = librosa.load(path, sr=sr)
    audio = audio_full[:audio_length]
    audio = torch.from_numpy(audio).reshape(1, 1, -1) / audio.max()
    return audio


def process_audio(audio):
    audio = audio.to(device)
    z = emb_model.encode(audio)
    cqt = blender.time_transform(audio)
    cqt = torch.nn.functional.interpolate(cqt,
                                          size=(z.shape[-1]),
                                          mode="nearest")
    cqt = (cqt - torch.min(cqt)) / (torch.max(cqt) - torch.min(cqt) + 1e-4)
    return z, cqt

def genrerate(
        path1 = './data/piano_guitar_1.wav',
        path2 = './data/piano_guitar_1_target.wav',
        nb_steps = 40,  #Number of diffusion steps
        guidance = 2.0  #Classifier free guidance strength
):
    """
    path1: input audio
    path2: target timbre
    """
    # Compute embeddings and CQT
    audio1, audio2 = load_audio(path1, sr=SR), load_audio(path2, sr=SR)
    z1, cqt1 = process_audio(audio1)
    z2, cqt2 = process_audio(audio2)

    # Compute structure representation
    time_cond1, time_cond2 = blender.encoder_time(cqt1), blender.encoder_time(cqt2)

    # Compute timbre representation
    zsem1, zsem2 = blender.encoder(z1), blender.encoder(z2)

    # Sample initial noise
    x0 = torch.randn_like(z1)

    # Timbre of sample 2 and structure of sample 1
    xS = blender.sample(x0,
                        time_cond=time_cond1,
                        zsem=zsem2,
                        nb_step=nb_steps,
                        guidance=guidance,
                        guidance_type="time_cond")

    audio_out = emb_model.decode(xS).cpu().numpy().squeeze()

    return audio_out,  SR

if __name__ == "__main__":
    main()
