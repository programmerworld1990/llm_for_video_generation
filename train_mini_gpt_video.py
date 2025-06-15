import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from mini_gpt import MiniGPT, MiniGPTConfig

# Clear GPU memory
torch.cuda.empty_cache()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mini_gpt_tokenizer")

# Load dataset from pickle
with open("video_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Debug: Verify loaded data
print("Loaded dataset keys:", list(data.keys()))
print("Sample 0 video shape:", data["video"][0].shape)
print("Sample 0 video type:", type(data["video"][0]))
print("Sample 0 text:", data["text"][0])
print("Sample 0 video min/max:", data["video"][0].min(), data["video"][0].max())

# Custom dataset class
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
        sample = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "video": video
        }
        return sample

# Create dataset
dataset = VideoDataset(data["text"], data["video"], tokenizer)

# Debug: Verify dataset
sample = dataset[0]
print("Dataset sample keys:", sample.keys())
print("Sample 0 video shape:", sample["video"].shape)
print("Sample 0 video type:", type(sample["video"]))
print("Sample 0 video min/max:", sample["video"].min().item(), sample["video"].max().item())

# Custom data collator
def video_data_collator(features):
    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "video": torch.stack([f["video"] for f in features])
    }
    return batch

# Initialize model
config = MiniGPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_embd=256,
    n_layer=2,
    n_head=4,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    frame_size=(1, 32, 64)
)
model = MiniGPT(config).to("cuda" if torch.cuda.is_available() else "cpu")

# Training arguments
training_args = TrainingArguments(
    output_dir="./mini_gpt_video_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    eval_strategy="steps",
    eval_steps=1000,
    max_grad_norm=1.0,
    gradient_accumulation_steps=1,
    dataloader_num_workers=0,
    dataloader_drop_last=False,
)

# Custom trainer
class VideoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            video_frames=inputs["video"],
            attention_mask=inputs["attention_mask"],
            mode="video"
        )
        loss = outputs["loss"]
        if self.state.global_step % 100 == 0:
            print(f"Step {self.state.global_step}, Logits min/max: {outputs['logits'].min().item():.4f}/{outputs['logits'].max().item():.4f}")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # Rename 'video' to 'video_frames'
            inputs["video_frames"] = inputs.pop("video")
            outputs = model(
                input_ids=inputs["input_ids"],
                video_frames=inputs["video_frames"],
                attention_mask=inputs["attention_mask"],
                mode="video"
            )
            loss = outputs["loss"]
            logits = outputs["logits"]
        return (loss, logits, None)  # No labels needed for MSE loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=video_data_collator,
            num_workers=self.args.dataloader_num_workers,
            drop_last=self.args.dataloader_drop_last
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=video_data_collator,
            num_workers=self.args.dataloader_num_workers,
            drop_last=self.args.dataloader_drop_last
        )

# Initialize trainer
trainer = VideoTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=video_data_collator
)

# Train
trainer.train()

# Save model
model = model.cpu()  # Move to CPU before saving
model.save_pretrained("./mini_gpt_video_model")
tokenizer.save_pretrained("./mini_gpt_video_model")
print("Model and tokenizer saved to ./mini_gpt_video_model")