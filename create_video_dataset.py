import numpy as np
import pickle

# Generate synthetic video dataset
num_samples = 1000
num_frames = 25  # For 5 seconds at 0.2s per frame
height, width = 32, 64  # New frame size
channels = 1  # Grayscale
videos = []
texts = []

for i in range(num_samples):
    # Create blank video
    video = np.zeros((num_frames, channels, height, width), dtype=np.float32)
    # Simulate object (4x4 square) moving right or changing intensity
    x = np.random.randint(4, width - 8)  # Starting x position
    y = np.random.randint(4, height - 8)  # Starting y position
    intensity = np.random.uniform(0.5, 0.8)  # Starting intensity
    for t in range(num_frames):
        # Clear frame
        video[t, 0] = 0
        # Place object
        video[t, 0, y:y+4, x:x+4] = intensity
        x = min(x + 1, width - 4)  # Move right
        intensity = min(intensity + 0.02, 1.0)  # Brighten slightly
    videos.append(video)
    texts.append(f"Square moves right and brightens in frame {np.random.randint(1, num_frames)}")

# Save dataset
dataset = {"text": texts, "video": videos}
with open("video_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

# Debug: Verify dataset
print("Before saving:")
print("Dataset keys:", list(dataset.keys()))
print("Sample 0 video shape:", dataset["video"][0].shape)
print("Sample 0 video type:", type(dataset["video"][0]))
print("Sample 0 text:", dataset["text"][0])
print("Sample 0 video min/max:", dataset["video"][0].min(), dataset["video"][0].max())

# Reload to confirm
with open("video_dataset.pkl", "rb") as f:
    loaded_dataset = pickle.load(f)
print("\nAfter reloading:")
print("Dataset keys:", list(loaded_dataset.keys()))
print("Sample 0 video shape:", loaded_dataset["video"][0].shape)
print("Sample 0 video type:", type(loaded_dataset["video"][0]))
print("Sample 0 video min/max:", loaded_dataset["video"][0].min(), loaded_dataset["video"][0].max())