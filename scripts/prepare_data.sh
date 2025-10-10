#!/bin/bash

# Create the processed spectrograms directory if it doesn't exist
mkdir -p ../data/processed/spectrograms

# Convert audio files to spectrograms for each species
for species in species_1 species_2 species_3; do
    for audio_file in ../data/raw/$species/*.wav; do
        # Extract the filename without extension
        filename=$(basename "$audio_file" .wav)
        # Generate the spectrogram image
        python ../src/data_prep/audio_to_spectrogram.py "$audio_file" "../data/processed/spectrograms/${species}_${filename}.png"
    done
done

echo "Data preparation complete. Spectrograms are saved in ../data/processed/spectrograms."