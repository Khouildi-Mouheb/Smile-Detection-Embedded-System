import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataEngine.dataSetBuilder import DataSetBuilder
import matplotlib.pyplot as plt

# Paths to your data files
path_to_train = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'train_happy.h5')
path_to_test = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'test_happy.h5')

# Load the dataset
train_x, train_y, test_x, test_y = DataSetBuilder.load_dataset(path_to_train, path_to_test)

# Visualize a few training images
num_samples_to_show = 5
for i in range(num_samples_to_show):
    save_path = f'train_sample_{i}.png'
    DataSetBuilder.show_sample(train_x, train_y, i, save_path=save_path)

# Visualize a few test images
for i in range(num_samples_to_show):
    save_path = f'test_sample_{i}.png'
    DataSetBuilder.show_sample(test_x, test_y, i, save_path=save_path)

print("Visualization complete. Images saved as PNG files.")