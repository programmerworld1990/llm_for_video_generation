import torch
from transformers import AutoTokenizer
from mini_gpt import MiniGPT
from PIL import Image
import imageio
import numpy as np

# Load model and tokenizer
model = MiniGPT.from_pretrained("./mini_gpt_video_model").to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./mini_gpt_video_model")
model.eval()  # Set model to evaluation mode

# Generate video
prompt = "Square moves right and brightens in frame 1"
inputs = tokenizer(prompt, return_tensors="pt", max_length=10, padding="max_length").to(model.device)

# Prepare dummy video frames as input (batch_size, num_frames, channels, height, width)
num_frames = 25  # For 5 seconds at 0.2s per frame
batch_size = 1
channels, height, width = model.config.frame_size  # (1, 32, 64)
dummy_frames = torch.zeros(batch_size, num_frames, channels, height, width).to(model.device)

# Generate frames using forward pass
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        video_frames=dummy_frames,
        attention_mask=inputs["attention_mask"],
        mode="video"
    )
    frames = outputs["logits"]  # [1, 25, 1, 32, 64]
    print("Logits min/max:", frames.min().item(), frames.max().item())  # Debug

# Save as GIF
frames = frames.squeeze(0).cpu().numpy()  # [25, 1, 32, 64]
print("Frames min/max before clip:", frames.min(), frames.max())  # Debug
frames = np.clip(frames, 0, 1)  # Ensure values in [0, 1]
frames = (frames * 255).astype(np.uint8)  # Scale to 0-255
print("Frames min/max after scale:", frames.min(), frames.max())  # Debug
gif_frames = []
for frame in frames:
    img = Image.fromarray(frame.squeeze(0))  # [32, 64]
    img = img.resize((512, 256), Image.LANCZOS)  # Upscale to 256x512
    gif_frames.append(img)
imageio.mimsave("generated_video.gif", gif_frames, duration=0.2)
print("Saved generated_video.gif")