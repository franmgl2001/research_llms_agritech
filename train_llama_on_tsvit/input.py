import torch
from torch import nn
from transformers import AutoProcessor, MllamaForConditionalGeneration

# 1) CONFIG
model_id  = "meta-llama/Llama-3.2-11B-Vision-Instruct"
emb_path  = "embeddings/tensor_219.pt"   # your saved [B, seq_vis, 128] tensor
prompt    = "<image> I'm going to pass you the embeddings of a image time series from a crop prediction model. To describe me what crop was sown, and when it got harvested..."

# 2) LOAD MODEL & PROCESSOR
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
# 🔧 CORRECTION #1: tie weights to avoid the "weights are not tied" warning
model.tie_weights()

device = next(model.parameters()).device

# 3) LOAD & MAP YOUR EMBEDDINGS
x_tensor = torch.load(emb_path, map_location=device).half()
print("Loaded tensor shape:", x_tensor.shape)   # e.g. [19, 144, 128]

# up-project 128 → projector.in_features (4096)
mapper = nn.Linear(
    x_tensor.size(-1),                           # 128
    model.multi_modal_projector.in_features      # 4096
).to(device).half()
mapped = mapper(x_tensor)                        # [B, seq_vis, 4096]
print("Mapped tensor shape:", mapped.shape)

proj = model.multi_modal_projector(mapped)       # [B, seq_vis, hidden_sz]
cross_attention_states = proj.reshape(
    proj.size(0),                                # B
    proj.size(1),                                # seq_vis
    model.config.text_config.hidden_size         # hidden_sz
)

# 4) TOKENIZE PROMPT
enc = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
)
input_ids      = enc.input_ids.to(device)
text_attn_mask = enc.attention_mask.to(device)

# 🔧 CORRECTION #2: build a 4-D cross_attention_mask
# according to model_doc: (batch, seq_length, max_num_images=1, max_num_tiles=seq_vis)
batch_size, text_len = input_ids.shape
seq_vis = x_tensor.shape[1]
cross_attention_mask = torch.ones(
    batch_size,
    text_len,
    1,        # single image
    seq_vis,  # number of vision tokens
    dtype=torch.long,
    device=device,
)

# 5) GENERATE
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=text_attn_mask,
    cross_attention_states=cross_attention_states,
    cross_attention_mask=cross_attention_mask,
    max_new_tokens=64,
    do_sample=False,
)

# 6) DECODE & PRINT
output = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True
)
print("→", output)
