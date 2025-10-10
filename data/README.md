# Data Structure and Contents

This project is designed to classify bat species using audio recordings. The data directory is structured as follows:

## Directory Structure

- **data/raw**: Contains subdirectories for each bat species, where the raw audio files are stored.
  - **species_1**: Raw audio files for species 1.
  - **species_2**: Raw audio files for species 2.
  - **species_3**: Raw audio files for species 3.

- **data/processed**: Contains processed data.
  - **spectrograms**: This subdirectory will store the spectrogram images generated from the raw audio files.

## Data Preparation

The raw audio files will be converted into spectrogram images using the scripts provided in the `src/data_prep` directory. The processed spectrograms will be used for training the deep learning model.

## Usage

To prepare the data, run the `prepare_data.sh` script located in the `scripts` directory. This will automate the conversion of audio files to spectrograms and organize them into the appropriate directories.

Ensure that you have the necessary dependencies installed as specified in the `requirements.txt` file before running the data preparation script.