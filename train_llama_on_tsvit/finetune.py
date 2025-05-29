import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, MllamaForConditionalGeneration

# 1) CONFIGURATION
MODEL_ID   = "meta-llama/Llama-3.2-11B-Vision-Instruct"
EMBED_LIST = "embeddings/tensor_list.jsonl" # Make sure this path is correct
print(torch.backends.mps.is_available())     # Should be True

DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 3
LR         = 1e-5  # Reduced learning rate
MAX_GRAD_NORM = 1.0  # Add gradient clipping
TARGET_NUM_VISION_TOKENS = 1601

print(f"Using device: {DEVICE}")
print(f"Targeting {TARGET_NUM_VISION_TOKENS} visual tokens.")

# 2) LOAD MODEL & PROCESSOR
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
model.tie_weights()
model.to(DEVICE)

# 3) DEFINE TRAINABLE MAPPER & FREEZE BASE MODEL
mapper_input_dim = 128
mapper = nn.Linear(
    mapper_input_dim,
    model.multi_modal_projector.in_features,
).to(DEVICE).half()

for p in model.parameters(): p.requires_grad = False
for p in mapper.parameters(): p.requires_grad = True

# 4) DATASET DEFINITION
class EmbeddingDataset(Dataset):
    def __init__(self, jsonl_path):
        self.items = [json.loads(line) for line in open(jsonl_path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        emb = torch.load(item['path'], map_location='cpu').to(torch.float16)

        num_total_patches = emb.shape[0] * emb.shape[1] # Should be 2736
        emb_flattened = emb.view(num_total_patches, emb.shape[2]) # [2736, 128]

        mapped_features = mapper(emb_flattened.to(DEVICE))      # [2736, proj_in_features]
        projected_features = model.multi_modal_projector(mapped_features) # [2736, model_hidden_size]

        # Adjust sequence length from 2736 to TARGET_NUM_VISION_TOKENS (1601)
        current_seq_len = projected_features.shape[0]
        if current_seq_len == TARGET_NUM_VISION_TOKENS:
            final_projected_features = projected_features
        elif current_seq_len > TARGET_NUM_VISION_TOKENS:
            # Interpolate: Reshape to [1, H, current_seq_len] for interpolate function
            p_reshaped = projected_features.permute(1, 0).unsqueeze(0)
            p_interpolated = torch.nn.functional.interpolate(
                p_reshaped,
                size=TARGET_NUM_VISION_TOKENS,
                mode='linear',
                align_corners=False
            )
            # Reshape back to [TARGET_NUM_VISION_TOKENS, H]
            final_projected_features = p_interpolated.squeeze(0).permute(1, 0)
            # Add small epsilon to prevent exact zeros
            final_projected_features = final_projected_features + 1e-6
        else:
            # Pad with small random values instead of zeros
            padding_size = TARGET_NUM_VISION_TOKENS - current_seq_len
            padding_tensor = torch.randn(
                padding_size,
                projected_features.shape[1],
                dtype=projected_features.dtype,
                device=projected_features.device
            ) * 1e-6
            final_projected_features = torch.cat((projected_features, padding_tensor), dim=0)
        
        # Shape: [1 (item_batch), 1 (num_images), TARGET_NUM_VISION_TOKENS, H]
        current_cross_states = final_projected_features.unsqueeze(0).unsqueeze(0)

        enc = tokenizer(
            item['prompt'], return_tensors='pt', padding=True, truncation=True
        )
        input_ids = enc.input_ids.squeeze(0)
        attn_mask = enc.attention_mask.squeeze(0)

        return input_ids, attn_mask, current_cross_states

# 5) COLLATE FUNCTION (remains the same as your last correct version)
def collate_fn(batch):
    ids, text_masks, visual_states_list_4d = zip(*batch)

    input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(DEVICE)
    attention_mask_for_text = pad_sequence(text_masks, batch_first=True, padding_value=0).to(DEVICE)
    
    if visual_states_list_4d[0].device != DEVICE:
         cross_states = torch.cat([s.to(DEVICE) for s in visual_states_list_4d], dim=0)
    else:
         cross_states = torch.cat(visual_states_list_4d, dim=0)

    B, T_text = input_ids.shape
    num_images_in_sequence = cross_states.shape[1] # This will be 1
    
    # cross_attn_mask shape: [B, T_text, num_images_in_sequence, 1] e.g. [4, T, 1, 1]
    cross_attn_mask = torch.ones(B, T_text, num_images_in_sequence, 1, dtype=torch.long, device=DEVICE)

    return input_ids, attention_mask_for_text, cross_states, cross_attn_mask

# 6) DATALOADER (remains the same)
dataset = EmbeddingDataset(EMBED_LIST)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0) 

# 7) OPTIMIZER (remains the same)
params_to_optimize = list(mapper.parameters())
optimizer = optim.Adam(params_to_optimize, lr=LR)

# 8) TRAIN LOOP (remains the same)
model.train()
print("Starting training loop...")
for epoch in range(NUM_EPOCHS):
    total_epoch_loss = 0.0
    for batch_idx, (input_ids, text_self_attention_mask, cross_attention_visual_states, cross_attention_visual_mask) in enumerate(loader):
        outputs = model(
            input_ids=input_ids,
            attention_mask=text_self_attention_mask,
            cross_attention_states=cross_attention_visual_states, # Now [B, 1, 1601, H]
            cross_attention_mask=cross_attention_visual_mask,   # Still [B, T, 1, 1]
            labels=input_ids,
        )
        loss = outputs.loss
        
        if torch.isnan(loss):
            print(f"NaN loss detected at Epoch {epoch+1}, Batch {batch_idx+1}. Stopping.")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(params_to_optimize, MAX_GRAD_NORM)
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in params_to_optimize:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
        
        if not has_nan_grad:
            optimizer.step()
        else:
            print(f"NaN gradients detected at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping update.")
            continue
        
        total_epoch_loss += loss.item()

    avg_epoch_loss = total_epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} avg loss {avg_epoch_loss:.4f}")

# 9) SAVE MAPPER (remains the same)
torch.save(mapper.state_dict(), "mapper_checkpoint.pt")
print("Done. Saved mapper_checkpoint.pt")