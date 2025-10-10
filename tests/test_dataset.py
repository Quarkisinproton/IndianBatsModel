import unittest
from src.datasets.spectrogram_dataset import SpectrogramDataset

class TestSpectrogramDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = SpectrogramDataset('data/processed/spectrograms', transform=None)

    def test_length(self):
        self.assertEqual(len(self.dataset), expected_length)  # Replace expected_length with the actual expected length

    def test_get_item(self):
        sample = self.dataset[0]
        self.assertIsNotNone(sample)  # Ensure that a sample is returned
        self.assertIn('image', sample)  # Check if the sample contains an image
        self.assertIn('label', sample)  # Check if the sample contains a label

    def test_labels(self):
        labels = [self.dataset[i]['label'] for i in range(len(self.dataset))]
        unique_labels = set(labels)
        self.assertEqual(unique_labels, {'species_1', 'species_2', 'species_3'})  # Replace with actual species labels

if __name__ == '__main__':
    unittest.main()