# Import required libraries
import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network modules (e.g., Linear, Conv2d)
from transformers import PretrainedConfig, PreTrainedModel  # Hugging Face base classes for model configuration and pretrained models

# Define configuration class for MiniGPT, inheriting from Hugging Face's PretrainedConfig
class MiniGPTConfig(PretrainedConfig):
    # Set model type identifier for Hugging Face serialization
    model_type = "mini_gpt"

    # Initialize configuration with default hyperparameters
    def __init__(
        self,
        vocab_size=50257,  # Size of text vocabulary (e.g., for tokenizer)
        n_positions=128,  # Maximum sequence length for text or video frames
        n_embd=256,  # Embedding dimension (increased to utilize more GPU memory)
        n_layer=2,  # Number of transformer layers
        n_head=4,  # Number of attention heads in transformer
        pad_token_id=0,  # ID for padding token in text sequences
        bos_token_id=1,  # ID for beginning-of-sequence token
        eos_token_id=2,  # ID for end-of-sequence token
        frame_size=(1, 32, 64),  # Video frame size (channels, height, width), e.g., 1x32x64 for grayscale
        **kwargs  # Additional arguments for parent class
    ):
        # Assign configuration parameters to instance variables
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.frame_size = frame_size
        # Initialize parent class (PretrainedConfig) with any additional arguments
        super().__init__(**kwargs)

# Define MiniGPT model, inheriting from Hugging Face's PreTrainedModel
class MiniGPT(PreTrainedModel):
    # Associate model with MiniGPTConfig
    config_class = MiniGPTConfig

    # Initialize model architecture
    def __init__(self, config):
        # Initialize parent class (PreTrainedModel) with configuration
        super().__init__(config)
        # Store configuration for access within model
        self.config = config
        # Embedding layer for text tokens: maps token IDs to vectors of size n_embd
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Learnable positional embeddings for sequences (text or video frames): [1, n_positions, n_embd]
        self.position_embedding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        # 2D convolutional layer for video frames: converts frame_size[0] channels to n_embd feature maps
        self.frame_conv = nn.Conv2d(config.frame_size[0], config.n_embd, kernel_size=3, padding=1)
        # List of transformer encoder layers: processes sequences with self-attention
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=config.n_head)
            for _ in range(config.n_layer)
        ])
        # Layer normalization applied after transformer layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Linear layer for text output: maps n_embd to vocab_size for next-token prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # Linear layer for video output: maps n_embd to flattened frame size (channels * height * width)
        self.frame_head = nn.Linear(config.n_embd, config.frame_size[0] * config.frame_size[1] * config.frame_size[2])
        # Initialize weights and apply any post-initialization steps (from PreTrainedModel)
        self.post_init()

    # Forward pass: processes input text or video and computes loss/logits
    def forward(self, input_ids=None, video_frames=None, attention_mask=None, mode="text"):
        # Determine device (GPU/CPU) based on input tensors
        device = input_ids.device if input_ids is not None else video_frames.device
        # Get batch size from input tensors
        batch_size = input_ids.size(0) if input_ids is not None else video_frames.size(0)
        # Get sequence length (text tokens or video frames)
        seq_len = input_ids.size(1) if input_ids is not None else video_frames.size(1)

        # Text mode: process tokenized text input
        if mode == "text":
            # Validate that input_ids is provided
            if input_ids is None:
                raise ValueError("input_ids must be provided in text mode")
            # Embed token IDs: [batch_size, seq_len] -> [batch_size, seq_len, n_embd]
            token_emb = self.token_embedding(input_ids)
            # Slice positional embeddings to match sequence length: [1, seq_len, n_embd]
            position_emb = self.position_embedding[:, :seq_len, :]
            # Add token and positional embeddings element-wise
            x = token_emb + position_emb  # [batch_size, seq_len, n_embd]

        # Video mode: process video frame input
        elif mode == "video":
            # Validate that video_frames is provided
            if video_frames is None:
                raise ValueError("video_frames must be provided in video mode")
            # Unpack video frame dimensions: [batch_size, num_frames, channels, height, width]
            b, t, c, h, w = video_frames.shape
            # Flatten batch and frame dimensions for convolution: [batch_size * num_frames, channels, height, width]
            video_frames_flat = video_frames.view(b * t, c, h, w)
            # Apply convolution to extract features: [batch_size * num_frames, n_embd, height, width]
            frame_emb = self.frame_conv(video_frames_flat)
            # Average pool over spatial dimensions: [batch_size * num_frames, n_embd]
            frame_emb = frame_emb.mean(dim=[2, 3])
            # Reshape to restore batch and frame dimensions: [batch_size, num_frames, n_embd]
            frame_emb = frame_emb.view(b, t, self.config.n_embd)
            # Slice positional embeddings to match number of frames: [1, num_frames, n_embd]
            position_emb = self.position_embedding[:, :t, :]
            # Add frame and positional embeddings element-wise
            x = frame_emb + position_emb  # [batch_size, num_frames, n_embd]

        # Invalid mode: raise error
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Pass through transformer layers
        for layer in self.transformer:
            # Apply transformer encoder layer (self-attention + feedforward): [batch_size, seq_len, n_embd]
            x = layer(x)
        # Apply final layer normalization
        x = self.ln_f(x)  # [batch_size, seq_len, n_embd]

        # Text mode output: predict next token
        if mode == "text":
            # Map embeddings to vocabulary logits: [batch_size, seq_len, vocab_size]
            logits = self.lm_head(x)
            # Compute cross-entropy loss for language modeling (flatten for loss calculation)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), input_ids.view(-1))

        # Video mode output: predict video frames
        else:
            # Map embeddings to flattened frame logits: [batch_size, num_frames, channels * height * width]
            logits = self.frame_head(x)
            # WARNING: Scaling logits by 1000 may cause large losses or saturated outputs
            # Consider removing or adjusting based on dataset/loss scaling
            logits = logits * 1000
            # Reshape logits to match frame dimensions: [batch_size, num_frames, channels, height, width]
            logits = logits.view(batch_size, t, *self.config.frame_size)
            # Compute mean squared error loss between predicted and target frames
            loss = nn.MSELoss()(logits, video_frames)

        # Return dictionary with loss and logits
        return {"loss": loss, "logits": logits}