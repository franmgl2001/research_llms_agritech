import torch

def data_collator(features, model):
    B = len(features)
    input_ids      = torch.stack([f["input_ids"]      for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels         = torch.stack([f["labels"]         for f in features])
    # emb comes in either [B, seq_vis, emb_dim] or [B, C, seq_vis, emb_dim]
    emb = torch.stack([f["embeddings"] for f in features])
    if emb.dim() == 3:
        emb = emb.unsqueeze(1)  # → [B, 1, seq_vis, emb_dim]
    B, C, seq_vis, emb_dim = emb.shape

    # Project each [seq_vis, emb_dim] chunk in-place
    # by reshaping into the model's expected 4-D pipeline
    # Step 1: flatten to apply your linear mapper + projector
    flat_seq = C * seq_vis
    emb_flat = emb.view(B, flat_seq, emb_dim)            # [B, C*seq_vis, emb_dim]
    mapped   = model.mapper(emb_flat)                    # [B, C*seq_vis, proj_in]
    proj     = model.multi_modal_projector(mapped)       # [B, C*seq_vis, hidden]

    # Step 2: un-flatten back into 4-D so the model can re-flatten correctly
    hidden = model.config.text_config.hidden_size
    cross_states = proj.view(B, C, seq_vis, hidden)      # [B, C, seq_vis, hidden]

    # Build a 4-D mask [B, text_len, C, seq_vis]
    text_len = input_ids.size(1)
    cross_attention_mask = torch.ones(
        B, text_len, C, seq_vis,
        dtype=torch.long, device=cross_states.device
    )

    return {
        "input_ids":              input_ids,
        "attention_mask":         attention_mask,
        "cross_attention_states": cross_states,      # 4-D
        "cross_attention_mask":   cross_attention_mask,  # 4-D
        "labels":                 labels,
    } 