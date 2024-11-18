import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Paths for the Monet and photo datasets
monet_path = './monet_jpg/'
photo_path = './photo_jpg/'

# Image transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize images to 256x256
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

class MonetDataset(Dataset):
    """Custom dataset class for loading Monet and photo images with transformations."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels (RGB)
        if self.transform:
            image = self.transform(image)
        return image

# Initialize datasets
monet_dataset = MonetDataset(monet_path, transform=transform)
photo_dataset = MonetDataset(photo_path, transform=transform)

# Create data loaders for batch processing
monet_loader = DataLoader(monet_dataset, batch_size=16, shuffle=True)
photo_loader = DataLoader(photo_dataset, batch_size=16, shuffle=True)

# --- Exploratory Data Analysis (EDA) ---

def show_sample_images(dataset, title="Sample Images"):
    """Display sample images from the dataset."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        image = dataset[i]
        image = image.permute(1, 2, 0)  # Change tensor format for plotting (C, H, W) -> (H, W, C)
        image = image * 0.5 + 0.5  # Unnormalize
        axes[i].imshow(image)
        axes[i].axis("off")
    fig.suptitle(title, fontsize=16)
    plt.show()

def calculate_statistics(dataset, title="Dataset Statistics"):
    """Calculate and display dataset statistics such as mean and std deviation of pixel intensities."""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images in DataLoader(dataset, batch_size=64):
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])

    mean /= len(dataset) / 64
    std /= len(dataset) / 64

    print(f"{title}:\nMean: {mean}\nStandard Deviation: {std}")

def explore_dataset(dataset, title="Dataset Exploration"):
    """Explore dataset properties such as size and first few image paths."""
    print(f"Exploring {title}...")
    print(f"Number of Images: {len(dataset)}")
    print(f"Sample Paths: {dataset.image_paths[:5]}")
    print("-" * 40)

# --- Main Function ---

if __name__ == "__main__":
    # Dataset Exploration
    explore_dataset(monet_dataset, title="Monet Dataset")
    explore_dataset(photo_dataset, title="Photo Dataset")

    # Display Sample Images
    print("Displaying Monet Sample Images...")
    show_sample_images(monet_dataset, title="Monet Images")

    print("Displaying Photo Sample Images...")
    show_sample_images(photo_dataset, title="Photo Images")

    # Calculate and Display Statistics
    print("Calculating Statistics for Monet Images...")
    calculate_statistics(monet_dataset, title="Monet Images Statistics")

    print("Calculating Statistics for Photo Images...")
    calculate_statistics(photo_dataset, title="Photo Images Statistics")
