import numpy as np

class ExampleDatasetGenerator:
    def __init__(self, image_height=640, image_width=640, num_channels=1, dataset_size=10):
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.dataset_size = dataset_size

    def generate_dataset(self):
        dataset = []
        for _ in range(self.dataset_size):
            image = np.random.randint(0, 256, (self.num_channels, self.image_height, self.image_width), dtype=np.uint8)
            dataset.append(image)
        return dataset

# Example usage:
generator = ExampleDatasetGenerator(num_channels=84, dataset_size=10)
dataset = generator.generate_dataset()
print(f"Generated dataset with {len(dataset)} images.")
