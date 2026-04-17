import h5py
import numpy as np
import matplotlib.pyplot as plt


class DataSetBuilder:
    def __init__(self):
        pass

    @staticmethod
    def load_dataset(path_to_train, path_to_test):
        with h5py.File(path_to_train, 'r') as train_dataset:
            train_x = np.array(train_dataset['train_set_x'][:])
            train_y = np.array(train_dataset['train_set_y'][:])

        with h5py.File(path_to_test, 'r') as test_dataset:
            test_x = np.array(test_dataset['test_set_x'][:])
            test_y = np.array(test_dataset['test_set_y'][:])

        # y reshaped
        train_y = train_y.reshape((1, train_x.shape[0]))
        test_y = test_y.reshape((1, test_y.shape[0]))

        print("train_x shape: " + str(train_x.shape))
        print("train_y shape: " + str(train_y.shape))
        print("test_x shape: " + str(test_x.shape))
        print("test_y shape: " + str(test_y.shape))

        return train_x, train_y, test_x, test_y

    @staticmethod
    def show_sample(images, labels, index, save_path=None):
        """Visualizes a specific image from the dataset using matplotlib."""
        plt.imshow(images[index])
        # Squeeze labels to handle (1, n) shape and get scalar for the index
        label_value = np.squeeze(labels)[index]
        plt.title(f"Sample index: {index} | Label: {label_value}")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            print(f"Image saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    # Configuration for the test
    # Replace these strings with the actual paths to your HDF5 files
    SAMPLE_TEST_PATH = r"C:\\Users\\Moi\\Desktop\\smileDetectionEmbeddedSystem\\data\\test_happy.h5"
    SAMPLE_TRAIN_PATH = r"C:\\Users\\Moi\\Desktop\\smileDetectionEmbeddedSystem\\data\\train_happy.h5"

    print("--- Running DataSetBuilder Test ---")
    try:
        # Load the data
        train_x, train_y, test_x, test_y = DataSetBuilder.load_dataset(SAMPLE_TRAIN_PATH, SAMPLE_TEST_PATH)
        
        # Test visualization by showing the first image in the training set
        print("\nDisplaying sample index 0...")
        DataSetBuilder.show_sample(train_x, train_y, index=0)
    except Exception as e:
        print(f"\n[Test Failed]: {e}")
        print("Hint: Ensure the .h5 files exist at the specified paths and you are running from the project root.")
