import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer, SFTConfig


# Fast, stable inits and less frag during load
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("TRANSFORMERS_NO_MEMORY_WARMUP", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

max_seq_length = 12288
model_name = "Kwaipilot/KAT-Dev"

# Bind this process to its GPU so ranks don't all touch cuda:0 at load
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(local_rank)
    except Exception:
        pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
    offload_state_dict=True,
)

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
)

EOS_TOKEN = tokenizer.eos_token or ""

# Add FIM tokens if needed
USE_FIM = True
FIM_TOKENS = ["<fim_prefix>", "<fim_middle>", "<fim_suffix>"]
if USE_FIM:
    missing = [t for t in FIM_TOKENS if t not in (tokenizer.additional_special_tokens or [])]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass


def formatting_examples(examples):
    """
    Normalize datasets that only provide a single 'text' column.
    - Ensures each entry ends with EOS.
    - Keeps any existing FIM markers intact if present in text.
    """
    # Support datasets that provide either 'text' or 'content'
    raw_texts = examples.get("text")
    if raw_texts is None:
        raw_texts = examples.get("content", [])
    texts = []
    for content in raw_texts:
        if content is None:
            continue
        content = content.rstrip()
        # Only append EOS if the example doesn't already end with EOS or <|endoftext|>
        if not content.endswith(EOS_TOKEN) and not content.endswith("<|endoftext|>"):
            content += EOS_TOKEN
        texts.append(content)
    return {"text": texts}


def chunk_long_texts(examples):
    """
    Chunk texts that exceed max_seq_length.
    Preserves FIM formatting, only chunks CLM samples.
    """
    chunks = []
    max_chunk_tokens = max_seq_length - 1
    eos_id = tokenizer.eos_token_id

    for text in examples["text"]:
        if not text:
            continue
        
        # Tokenize to check length
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        # If within limit, keep as-is
        if len(ids) <= max_chunk_tokens:
            chunks.append(text)
            continue
        
        # Handle FIM samples - don't chunk them (would break format)
        if '<fim_prefix>' in text or '<fim_middle>' in text or '<fim_suffix>' in text:
            # Truncate FIM samples instead of chunking
            truncated_ids = ids[:max_chunk_tokens]
            
            # Try to end at a reasonable point
            decoded = tokenizer.decode(truncated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            
            # Ensure proper FIM closure if truncated
            if '<fim_middle>' in decoded and not decoded.endswith('<|endoftext|>'):
                decoded += '<|endoftext|>'
            
            chunks.append(decoded)
            continue
        
        # Chunk CLM samples
        for start in range(0, len(ids), max_chunk_tokens):
            part_ids = ids[start:start + max_chunk_tokens]
            if not part_ids:
                continue
            
            # Decode chunk
            chunk_text = tokenizer.decode(part_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            
            # Ensure EOS at end of each chunk
            if eos_id is not None:
                # Check if last token is EOS
                if part_ids[-1] != eos_id:
                    chunk_text = chunk_text.rstrip()
                    if not chunk_text.endswith(EOS_TOKEN):
                        chunk_text += EOS_TOKEN
            
            chunks.append(chunk_text)

    return {"text": chunks}


# Load dataset
print("Loading dataset...")
ds = load_dataset("archit11/hyperswitch-token-aware-cpt-fixed")

# Create train/eval split for evaluation loss
if isinstance(ds, DatasetDict):
    if "validation" in ds:
        train_raw, eval_raw = ds["train"], ds["validation"]
    elif "test" in ds:
        train_raw, eval_raw = ds["train"], ds["test"]
    else:
        split = ds["train"].train_test_split(test_size=0.01, seed=42)
        train_raw, eval_raw = split["train"], split["test"]
else:
    split = ds.train_test_split(test_size=0.01, seed=42)
    train_raw, eval_raw = split["train"], split["test"]

print(f"Original train size: {len(train_raw)} | eval size: {len(eval_raw)}")
print(f"Train columns: {train_raw.column_names}")

# Format samples (handles type-specific processing)
print("Formatting samples (train/eval)...")
train_ds = train_raw.map(
    formatting_examples,
    batched=True,
    # Drop all original columns; mapper returns only 'text'
    remove_columns=train_raw.column_names,
    desc="Formatting train"
)
eval_ds = eval_raw.map(
    formatting_examples,
    batched=True,
    remove_columns=eval_raw.column_names,
    desc="Formatting eval"
)

# Chunk long sequences
print("Chunking long sequences (train/eval)...")
train_ds = train_ds.map(
    chunk_long_texts,
    batched=True,
    remove_columns=[],
    num_proc=1,
    desc="Chunking train"
)
eval_ds = eval_ds.map(
    chunk_long_texts,
    batched=True,
    remove_columns=[],
    num_proc=1,
    desc="Chunking eval"
)

print(f"Final sizes after chunking -> train: {len(train_ds)} | eval: {len(eval_ds)}")

# Verify samples
print("\nSample verification (train):")
sample = train_ds[0]
print(f"Sample text length: {len(sample['text'])} chars")
print(f"Sample tokens: {len(tokenizer.encode(sample['text']))}")
print(f"First 200 chars: {sample['text'][:200]}")

# Training config with stability improvements
batch_size_per_gpu = 1
grad_accum = 4
learning_rate = 2e-5  # Reduced from 5e-5 for stability
num_epochs = 2
warmup_steps = 50  # Using steps instead of ratio for more control

training_args = SFTConfig(
    gradient_accumulation_steps=grad_accum,
    per_device_train_batch_size=batch_size_per_gpu,
    per_device_eval_batch_size=batch_size_per_gpu,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    
    # Stability improvements
    max_grad_norm=0.5,  # Reduced from 1.0
    warmup_steps=warmup_steps,
    
    # Gradient checkpointing with more stable config
    gradient_checkpointing=False,
    #gradient_checkpointing_kwargs={"use_reentrant": False},
    
    bf16=True,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="outputs-hyperswitch-lora",
    report_to="wandb",
    save_strategy="steps",
    save_steps=40,
    eval_strategy="steps",
    eval_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,  # Important: packing can break FIM samples
    
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    
    ddp_find_unused_parameters=False,
    deepspeed="ds_zero3.json",
)

print("\n" + "="*60)
print("Training Configuration")
print("="*60)
print(f"  Model: {model_name}")
print(f"  Max sequence length: {max_seq_length}")
print(f"  Learning rate: {learning_rate:.2e}")
print(f"  Batch size per GPU: {batch_size_per_gpu}")
print(f"  Gradient accumulation: {grad_accum}")
print(f"  Effective batch size: {batch_size_per_gpu * grad_accum}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Max grad norm: {training_args.max_grad_norm}")
print(f"  Epochs: {num_epochs}")
print(f"  Dataset size: {len(train_ds)}")
print("="*60 + "\n")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
print("\nSaving model...")
model.save_pretrained("hyperswitch_lora")
tokenizer.save_pretrained("hyperswitch_lora")

print("✅ Training complete!")
