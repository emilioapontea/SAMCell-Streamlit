import cv2
import torch

model = None #! FinetunedSAM('facebook/sam-vit-base', finetune_vision=False, finetune_prompt=True, finetune_decoder=True) # FinetunedSAM instance
trained_samcell_path: str = None #! SAMCell weights
model.load_weights(trained_samcell_path)
pipeline = None #! SlidingWindowPipeline(model, 'cuda', crop_size=256) # CUDA pipeline