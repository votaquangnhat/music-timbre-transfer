import yaml
import pandas as pd
from midi2audio import FluidSynth
import pretty_midi
import os
import shutil
import random
import numpy as np

import audio_wrapper as ap

# load config
with open("config/data_config.yaml", "r") as f: # need to run in the project root dir
    CONFIG = yaml.safe_load(f)

def select_midi_files(
        maestro_metadata: pd.DataFrame,
        spectrogram_metadata: pd.DataFrame,
        choosed_subset: str, # either 'train', 'validation' or 'test'
):
    subset_metadata = maestro_metadata[maestro_metadata["split"] == choosed_subset]

    required_total_audio_length = CONFIG["num_images_to_create_per_timbre"][choosed_subset] * 7 # about 6 seconds for each image, plus 1 for safety
    timbres = CONFIG["timbres"]

    used_pairs = set(zip(spectrogram_metadata["maestro_id"], spectrogram_metadata["timbres"]))

    available_midi_ids = {
        timbre: [
            maestro_id for maestro_id in range(len(subset_metadata))
            if (maestro_id, timbre) not in used_pairs
        ]
        for timbre in timbres
    } # I will not use all of these

    selected_midi_ids = { timbre: [] for timbre in timbres }

    # for each timbre, choose the songs of which combined duration is at least required_total_audio_length
    for timbre in timbres:
        temp = available_midi_ids[timbre]
        random.Random(42).shuffle(temp) # replicable purpose

        total_audio_length = 0
        extra_after_reached = 5 # after reaching the required length, I will still take some extra songs
        extras_taken = 0
        reached_required_length = False

        while temp:
            id = temp.pop(0)
            duration = subset_metadata.iloc[id]["duration"]
            total_audio_length += duration
            selected_midi_ids[timbre].append(id)

            if reached_required_length:
                extras_taken += 1
                if extras_taken >= extra_after_reached:
                    break

            elif total_audio_length >= required_total_audio_length:
                reached_required_length = True
            
    return selected_midi_ids

def create_wav_file_from_midi(midi_file_path, output_path, timbre):
    wav_dir = CONFIG["wav_dir"]
    soundfont_num = CONFIG["soundfont_num"]

    soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    fs = FluidSynth(sound_font=soundfont_path)

    temp_midi_file_path = f'{wav_dir}/temp_song.mid'
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    for instrument in midi_data.instruments: # turn all instruments to the same timbre for consistency
        instrument.program = soundfont_num[timbre]
    midi_data.write(temp_midi_file_path)
    fs.midi_to_audio(temp_midi_file_path, output_path)
    print(f"Created {output_path}")
    os.remove(temp_midi_file_path)

def generate_spectrograms(
        selected_midi_ids: dict[str, list[int]],
        maestro_metadata: pd.DataFrame,
        spectrogram_metadata: pd.DataFrame,
        choosed_subset: str
):
    """
        Generate spectrograms in form of .npy files and save
    """
    timbres = CONFIG["timbres"]
    maestro_data_dir = CONFIG["maestro_data_dir"]
    wav_dir = CONFIG["wav_dir"]
    num_images_to_create_per_timbre = CONFIG["num_images_to_create_per_timbre"]
    spectrogram_dir = CONFIG["spectrogram_dir"]

    target = num_images_to_create_per_timbre[choosed_subset]
    
    # for each (timbre, id), create wav file, then create spectrograms
    for timbre in timbres:
        num_of_created_spectrograms = 0
        for id in selected_midi_ids[timbre]:
            if num_of_created_spectrograms >= target:
                break
            midi_file_path = f'{maestro_data_dir}/{maestro_metadata.iloc[id]["midi_filename"]}'
            audio_path = f'{wav_dir}/song_{id}_{timbre}.wav'
            create_wav_file_from_midi(midi_file_path=midi_file_path, output_path=audio_path, timbre=timbre)

            audio_wrapper = ap.AudioWrapper()
            audio_wrapper.load_audio(audio_path)

            for i in range(audio_wrapper.number_of_slices):
                if num_of_created_spectrograms >= target:
                    break
                file_name = f'spectrogram_{id}_slice_{i}_{timbre}.npy'
                save_path = f'{spectrogram_dir}/{file_name}'
                spectrogram = audio_wrapper.spectrogram_from_audio_slice(i)
                audio_wrapper.save_spectrogram(spectrogram, save_path)
                print(f"Created {save_path}")

                #add to metadata
                spectrogram_metadata.loc[len(spectrogram_metadata)] = {
                "maestro_id": id,
                "slice": i,
                "timbres": timbre,
                "file_name": file_name,
                "subset": choosed_subset
                }
                num_of_created_spectrograms += 1
                
    spectrogram_metadata.to_csv(f"{spectrogram_dir}/spectrogram_metadata.csv", index=False)

    # delete wav files to save space
    try:
        shutil.rmtree(wav_dir)
        os.makedirs(wav_dir, exist_ok=True)
    except Exception as e:
        print(f"Error while deleting wav files: {e}")


def main(
        choosed_subset = "train", # either 'train', 'validation' or 'test'
):
    # load metadata
    maestro_metadata = pd.read_csv(CONFIG["maestro_metadata_file"])
    spectrogram_metadata = pd.DataFrame()
    try:
        spectrogram_metadata = pd.read_csv(f"{CONFIG['spectrogram_dir']}/spectrogram_metadata.csv")
    except FileNotFoundError:
        spectrogram_metadata = pd.DataFrame(columns=["maestro_id", "slice", "timbres", "file_name", "subset"])
        print("No existing spectrogram metadata found. A new empty dataframe is created.")
    finally:
        print(f"Existing spectrogram metadata has {len(spectrogram_metadata)} rows.")

    # 1. select midi files for each timbre
    selected_midi_ids = select_midi_files(
        maestro_metadata=maestro_metadata,
        spectrogram_metadata=spectrogram_metadata,
        choosed_subset=choosed_subset
    )

    print(f"Selected midi ids for {choosed_subset} subset:")
    for timbre, ids in selected_midi_ids.items():
        print(f"{timbre}: {ids}")

    # 2. create spectrograms 
    generate_spectrograms(
        selected_midi_ids=selected_midi_ids,
        maestro_metadata=maestro_metadata,
        spectrogram_metadata=spectrogram_metadata,
        choosed_subset=choosed_subset
    )

    print(f'updated spectrogram metadata has {len(spectrogram_metadata)} rows.')

import time
if __name__ == '__main__':
    start_time = time.time()
    main(choosed_subset="train")
    main(choosed_subset="validation")
    main(choosed_subset="test")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
