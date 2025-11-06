import torch
import math
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import get_cosine_schedule_with_warmup


import os
import math

# os.environ["HF_ENDPOINT"] = "http://localhost:5564"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Disable Transformers' load-time memory warmup to avoid large cuda allocations
os.environ.setdefault("TRANSFORMERS_NO_MEMORY_WARMUP", "1")
# Help CUDA memory fragmentation during big model loads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
max_seq_length = 12288
# The FP8 variant requires GPUs with compute capability >= 8.9 (e.g., H100/4090).
# A100 is 8.0, so use a non-FP8 checkpoint.
model_name = "Kwaipilot/KAT-Dev"

# Respect local rank to avoid all ranks allocating on cuda:0 during load
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(local_rank)
    except Exception:
        pass

# Build a max_memory map so Transformers shards weights safely across GPUs
def build_max_memory(reserve_gib_per_gpu=2):
    if not torch.cuda.is_available():
        return None
    max_mem = {}
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allow = max(1, int(total - reserve_gib_per_gpu))
        max_mem[i] = f"{allow}GiB"
    return max_mem

max_memory = build_max_memory(reserve_gib_per_gpu=2)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None,  # DeepSpeed/Accelerate decide placement
    low_cpu_mem_usage=True,
    offload_state_dict=True,
)

# Enable LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
  #  use_gradient_checkpointing="unsloth",
)

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     args=training_args,
# )

# trainer.train()


EOS_TOKEN = tokenizer.eos_token

# CPT formatting: map dataset to plain text from 'content' or 'text'
def formatting_cpt(examples):
    texts = []
    if "content" in examples:
        headings = examples.get("heading") or [None] * len(examples["content"])
        for h, c in zip(headings, examples["content"]):
            prefix = (h + "\n\n") if h else ""
            texts.append((prefix + (c or "")).strip())
    elif "text" in examples:
        for t in examples["text"]:
            texts.append((t or "").strip())
    else:
        # Fallback: join available fields
        keys = list(examples.keys())
        for i in range(len(examples[keys[0]])):
            row = []
            for k in keys:
                row.append(str(examples[k][i]))
            texts.append(" \n".join(row).strip())
    return {"text": texts}


def chunk_long_texts(examples):
    """Split long documents so every chunk stays within the tokenizer context."""
    chunked_texts = []
    max_chunk_tokens = max_seq_length - 1
    eos_id = tokenizer.eos_token_id

    for text in examples["text"]:
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            if not token_ids or token_ids[-1] != eos_id:
                token_ids.append(eos_id)
        else:
            # Fall back to string EOS if tokenizer lacks eos_token_id.
            if EOS_TOKEN and not text.endswith(EOS_TOKEN):
                text = text + EOS_TOKEN
            chunked_texts.append(text)
            continue

        for start in range(0, len(token_ids), max_chunk_tokens):
            chunk_ids = token_ids[start : start + max_chunk_tokens]
            if not chunk_ids:
                continue
            if eos_id is not None and chunk_ids[-1] != eos_id:
                chunk_ids.append(eos_id)
            chunked_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            chunked_texts.append(chunked_text)

    return {"text": chunked_texts}

# Use your deepwiki dataset for CPT
ds_loaded = load_dataset("archit11/deepwiki-16k")
dataset = ds_loaded["train"] if isinstance(ds_loaded, dict) and "train" in ds_loaded else ds_loaded
dataset = dataset.map(formatting_cpt, batched=True, remove_columns=dataset.column_names)
dataset = dataset.map(chunk_long_texts, batched=True, num_proc=1)

# Fixed hyperparameters (edit these as needed)
batch_size_per_gpu = 2
grad_accum = 4
learning_rate = 5e-5
num_epochs = 2
warmup_ratio = 0.03

training_args = SFTConfig(
    gradient_accumulation_steps=grad_accum,
    per_device_train_batch_size=batch_size_per_gpu,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    output_dir="outputs-16k-lora",
    report_to="wandb",
    save_strategy="steps",
    save_steps=500,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    ddp_find_unused_parameters=False,
    deepspeed="ds_zero3.json",
    warmup_ratio=warmup_ratio,
    max_grad_norm=0.6,
)

print("Training hyperparameters:")
print(f"  Learning rate: {learning_rate:.2e}")
print(f"  Batch size per GPU: {batch_size_per_gpu}")
print(f"  Gradient accumulation: {grad_accum}")
print(f"  Effective batch size: {batch_size_per_gpu * grad_accum * 4}")
print(f"  Warmup ratio: {warmup_ratio}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()


model.save_pretrained("katdev_lora")
tokenizer.save_pretrained("katdev_lora")
