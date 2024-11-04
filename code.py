import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the main dataset directory
dataset_path = "/kaggle/input/gan-getting-started"

# Explore subdirectories
def explore_directories():
    """Explore and print the contents of each subdirectory in the dataset."""
    for subdir, _, files in os.walk(dataset_path):
        print(f"📂 Directory: {subdir}")
        print("  📄 Files:", files[:5], "..." if len(files) > 5 else "")
        print("-" * 40)

# Display a few sample images from each category
def display_sample_images(subfolder_name, num_samples=5):
    """Display a specified number of sample images from a given subfolder."""
    subfolder_path = os.path.join(dataset_path, subfolder_name)
    image_files = os.listdir(subfolder_path)[:num_samples]

    plt.figure(figsize=(15, 5))
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(subfolder_path, image_file)
        img = load_img(img_path, target_size=(256, 256))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{subfolder_name} Sample {i+1}")
    plt.show()

# Load and preprocess images
def load_images(subfolder_name, img_size=(256, 256)):
    """Load and preprocess images from a specified subfolder."""
    images = []
    subfolder_path = os.path.join(dataset_path, subfolder_name)
    for image_file in os.listdir(subfolder_path):
        img_path = os.path.join(subfolder_path, image_file)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
    return np.array(images)

# Define a simple GAN model
def build_gan(input_dim=(256, 256, 3)):
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
        tf.keras.layers.Reshape((16, 16, 1)),
        tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
    ])
    
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=input_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    return generator, discriminator

# Train GAN (dummy training loop for demonstration)
def train_gan(generator, discriminator, monet_images, epochs=1):
    batch_size = 16
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)
        
        # This is a dummy loop and would need real training logic.
        print(f"Epoch {epoch+1}/{epochs}: Dummy training...")

# Generate images using the trained GAN model
def generate_images(generator, num_images=7000, img_size=(256, 256)):
    """Generate images with the trained GAN model."""
    images = []
    for _ in range(num_images):
        noise = np.random.normal(0, 1, (1, 100))
        generated_img = generator.predict(noise)[0]
        images.append(generated_img)
    return images

# Save images to a zip file
def save_images_to_zip(images, zip_path="images.zip"):
    """Save a list of generated images into a zip file for submission."""
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, img_array in enumerate(images):
            img = tf.keras.preprocessing.image.array_to_img(img_array)
            img_filename = f"generated_image_{idx+1}.jpg"
            img.save(img_filename)
            zipf.write(img_filename)
            os.remove(img_filename)

# Main function to run all steps
if __name__ == "__main__":
    print("Exploring dataset directories...")
    explore_directories()

    # Display sample images from each category
    print("Displaying sample images from each subfolder...")
    display_sample_images("monet_jpg", num_samples=5)
    display_sample_images("photo_jpg", num_samples=5)

    # Load images for model training
    print("Loading Monet images for GAN training...")
    monet_images = load_images("monet_jpg")
    print(f"Loaded {len(monet_images)} Monet images.")

    # Build GAN model
    print("Building GAN model...")
    generator, discriminator = build_gan()

    # Train GAN model (This is a dummy training loop. Replace with actual training logic.)
    print("Training GAN model...")
    train_gan(generator, discriminator, monet_images, epochs=5)

    # Generate images
    print("Generating Monet-style images...")
    generated_images = generate_images(generator, num_images=7000)

    # Save generated images to zip for submission
    print("Saving generated images to images.zip...")
    save_images_to_zip(generated_images)

    print("Process complete. Submission file 'images.zip' is ready.")
