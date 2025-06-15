from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
video_frames = pipe("A bird flying over a forest", num_inference_steps=25).frames