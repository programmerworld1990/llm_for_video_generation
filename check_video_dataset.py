import pickle
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mini_gpt_tokenizer")

# Load dataset from pickle
with open("video_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Custom dataset class (same as train_mini_gpt_video.py)
class VideoDataset(Dataset):
    def __init__(self, texts, videos, tokenizer, max_length=10):
        self.texts = texts
        self.videos = [torch.tensor(v, dtype=torch.float32) for v in videos]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        video = self.videos[idx]
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "video": video
        }

# Create dataset
dataset = VideoDataset(data["text"], data["video"], tokenizer)

# Check all samples
for i in range(len(dataset)):
    sample = dataset[i]
    if "video" not in sample:
        print(f"Sample {i} missing 'video' key: {sample.keys()}")
    else:
        print(f"Sample {i} keys: {sample.keys()}, video shape: {sample['video'].shape}")
    if i >= 9:  # Check first 10 samples
        break