# IndianBatsModel
# Bat Species Classifier

This project aims to classify bat species using their vocalizations. The audio recordings are processed into spectrogram images, which are then used to train a deep learning model.

## Project Structure

- **data/**: Contains raw audio files and processed spectrogram images.
  - **raw/**: Subdirectories for each species (species_1, species_2, species_3) where the raw audio files are stored.
  - **processed/**: Contains a subdirectory named `spectrograms` for storing the processed spectrogram images.
  
- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis and visualization.
  - **exploratory.ipynb**: Used for analyzing the audio data and visualizing the spectrograms.

- **src/**: Contains the source code for data preparation, model training, and evaluation.
  - **data_prep/**: Scripts for preparing the data.
    - **audio_to_spectrogram.py**: Functions to convert audio recordings into spectrogram images.
    - **augment.py**: Functions for data augmentation techniques.
  - **datasets/**: Custom dataset class for loading and processing spectrogram images.
    - **spectrogram_dataset.py**: Defines the dataset class.
  - **models/**: Contains the model architecture.
    - **cnn.py**: Defines a convolutional neural network for classification.
  - **train.py**: Contains the training loop for the model.
  - **evaluate.py**: Functions to evaluate the trained model.
  - **utils.py**: Utility functions used throughout the project.

- **scripts/**: Contains scripts for automating tasks.
  - **prepare_data.sh**: Automates the data preparation process.

- **tests/**: Contains unit tests for the project.
  - **test_dataset.py**: Unit tests for the dataset class.

- **configs/**: Configuration settings for the project.
  - **config.yaml**: Contains hyperparameters and file paths.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **setup.py**: Used for packaging the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd bat-species-classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the data by running the shell script:
   ```
   bash scripts/prepare_data.sh
   ```

2. Train the model:
   ```
   python src/train.py
   ```

3. Evaluate the model:
   ```
   python src/evaluate.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.