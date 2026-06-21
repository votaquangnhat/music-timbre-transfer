import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from diffusers import StableDiffusionImg2ImgPipeline

import audio_wrapper as aw


def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference for Riffusion LoRA timbre transfer")

    # Normal audio input mode
    parser.add_argument("--input_audio", type=str, default=None)

    # Test-set input mode
    parser.add_argument("--from_test_set", action="store_true")
    parser.add_argument("--spectrograms_dir", type=str, default="data/spectrogram")
    parser.add_argument("--test_subset", type=str, default="test")
    parser.add_argument("--test_timbre", type=str, default="piano")
    parser.add_argument("--test_index", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="riffusion/riffusion-model-v1",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="outputs/test_lora_flute",
    )

    # Target
    parser.add_argument("--target_timbre", type=str, default="flute")
    parser.add_argument("--prompt", type=str, default=None)

    # Img2img settings
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Audio reconstruction
    parser.add_argument("--num_griffin_lim_iters", type=int, default=32)

    return parser.parse_args()


def save_audio(waveform: np.ndarray, sample_rate: int, path: str):
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    waveform = np.nan_to_num(waveform)

    peak = np.max(np.abs(waveform))
    if peak > 1.0:
        waveform = waveform / peak

    sf.write(
        file=str(path),
        data=waveform,
        samplerate=sample_rate,
        subtype="PCM_16",
    )


def load_from_audio_file(input_audio: str, converter: aw.SpectrogramConverter):
    audio_wrapper = aw.AudioWrapper()
    audio_wrapper.load_audio(input_audio)

    if audio_wrapper.number_of_slices >= 1:
        spectrogram = audio_wrapper.spectrogram_from_audio_slice(0)
    else:
        spectrogram = converter.spectrogram_from_waveform(audio_wrapper.audio)

    return spectrogram, audio_wrapper.sample_rate, Path(input_audio).name


def load_from_test_set(
    spectrograms_dir: str,
    test_subset: str,
    test_timbre: str,
    test_index: int,
):
    spectrograms_dir = Path(spectrograms_dir)
    metadata_path = spectrograms_dir / "spectrogram_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = {"file_name", "timbres", "subset"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    filtered = metadata[
        (metadata["subset"] == test_subset)
        & (metadata["timbres"] == test_timbre)
    ].reset_index(drop=True)

    if len(filtered) == 0:
        raise ValueError(
            f"No test samples found for subset={test_subset!r}, timbre={test_timbre!r}"
        )

    if test_index < 0 or test_index >= len(filtered):
        raise IndexError(
            f"test_index={test_index} out of range. "
            f"Available range: 0 to {len(filtered) - 1}"
        )

    row = filtered.iloc[test_index]
    file_name = row["file_name"]
    spectrogram_path = spectrograms_dir / file_name

    if not spectrogram_path.exists():
        raise FileNotFoundError(f"Spectrogram file not found: {spectrogram_path}")

    spectrogram = np.load(spectrogram_path)

    source_name = (
        f"test_{test_subset}_{test_timbre}_index_{test_index}_"
        f"{Path(file_name).stem}"
    )

    return spectrogram, source_name, row.to_dict(), len(filtered)


def main():
    args = parse_args()

    if not args.from_test_set and args.input_audio is None:
        raise ValueError("Please provide --input_audio, or use --from_test_set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    converter = aw.SpectrogramConverter()
    sample_rate = converter.sample_rate

    # -------------------------
    # 1. Load input spectrogram
    # -------------------------
    if args.from_test_set:
        spectrogram, source_name, metadata_row, total_candidates = load_from_test_set(
            spectrograms_dir=args.spectrograms_dir,
            test_subset=args.test_subset,
            test_timbre=args.test_timbre,
            test_index=args.test_index,
        )
        print(f"Loaded test-set sample: {source_name}")
        print(f"Total candidates for this filter: {total_candidates}")
        print(f"Metadata row: {metadata_row}")
    else:
        spectrogram, sample_rate, source_name = load_from_audio_file(
            input_audio=args.input_audio,
            converter=converter,
        )
        print(f"Loaded audio file: {source_name}")

    expected_shape = (converter.n_mels, converter.image_width)
    if spectrogram.shape != expected_shape:
        raise ValueError(
            f"Input spectrogram shape is {spectrogram.shape}, expected {expected_shape}. "
            "For normal audio mode, use a short audio that creates one 512x512 spectrogram."
        )

    init_image = converter.image_from_spectrogram(spectrogram)
    init_image.save(output_dir / "input_spectrogram.png")

    # Also save the input audio reconstructed from its spectrogram.
    # This is useful when using --from_test_set because there is no original wav path.
    input_waveform = converter.waveform_from_spectrogram(
        spectrogram,
        num_griffin_lim_iters=args.num_griffin_lim_iters,
        device=device,
    )
    save_audio(
        waveform=input_waveform,
        sample_rate=sample_rate,
        path=str(output_dir / "input.wav"),
    )

    # -------------------------
    # 2. Load base model + LoRA
    # -------------------------
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    print(f"Loading LoRA from: {args.lora_path}")
    pipe.load_lora_weights(args.lora_path)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # -------------------------
    # 3. Inference
    # -------------------------
    if args.prompt is None:
        prompt = f"a mel spectrogram of solo {args.target_timbre} music, same melody and rhythm"
    else:
        prompt = args.prompt

    generator = torch.Generator(device=device).manual_seed(args.seed)

    result_image = pipe(
        prompt=prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]

    result_image.save(output_dir / "output_spectrogram.png")

    # -------------------------
    # 4. Output spectrogram -> audio
    # -------------------------
    result_spectrogram = converter.spectrogram_from_image(result_image)
    result_waveform = converter.waveform_from_spectrogram(
        result_spectrogram,
        num_griffin_lim_iters=args.num_griffin_lim_iters,
        device=device,
    )

    save_audio(
        waveform=result_waveform,
        sample_rate=sample_rate,
        path=str(output_dir / "output.wav"),
    )

    print("\nDone.")
    print(f"Prompt: {prompt}")
    print(f"Input spectrogram:        {output_dir / 'input_spectrogram.png'}")
    print(f"Output spectrogram:       {output_dir / 'output_spectrogram.png'}")
    print(f"Input reconstructed wav:  {output_dir / 'input.wav'}")
    print(f"Output wav:               {output_dir / 'output.wav'}")


if __name__ == "__main__":
    main()