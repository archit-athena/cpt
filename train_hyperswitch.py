import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
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

# LoRA config (match existing)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
)

EOS_TOKEN = tokenizer.eos_token or ""

# Optional: enable Fill-in-the-Middle (FIM) formatting when dataset provides
# prefix/middle/suffix fields. We add common FIM tokens if missing.
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
    texts = []
    keys = set(examples.keys())

    # Prefer FIM when prefix/middle/suffix exist and USE_FIM is True
    if USE_FIM and {"prefix", "suffix"}.issubset(keys) and ("middle" in keys or "target" in keys):
        middles = examples.get("middle") or examples.get("target") or [None] * len(examples["prefix"])
        for pre, mid, suf in zip(examples["prefix"], middles, examples["suffix"]):
            pre = (pre or "").rstrip()
            mid = (mid or "").rstrip()
            suf = (suf or "").rstrip()
            # <fim_prefix> P <fim_suffix> S <fim_middle> M
            texts.append(f"<fim_prefix>{pre}<fim_suffix>{suf}<fim_middle>{mid}")
    elif "text" in examples:
        for t in examples["text"]:
            texts.append((t or "").rstrip())
    else:
        # Fallback: join visible columns into text
        order = list(keys)
        n = len(examples[order[0]])
        for i in range(n):
            row = []
            for k in order:
                row.append(str(examples[k][i]))
            texts.append("\n".join(row).rstrip())
    return {"text": texts}


def chunk_long_texts(examples):
    chunks = []
    max_chunk_tokens = max_seq_length - 1
    eos_id = tokenizer.eos_token_id

    for text in examples["text"]:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            if not ids or ids[-1] != eos_id:
                ids.append(eos_id)
        else:
            # Fall back to string eos
            chunks.append(text + (EOS_TOKEN if EOS_TOKEN and not text.endswith(EOS_TOKEN) else ""))
            continue

        for start in range(0, len(ids), max_chunk_tokens):
            part = ids[start:start + max_chunk_tokens]
            if not part:
                continue
            if eos_id is not None and part[-1] != eos_id:
                part.append(eos_id)
            chunks.append(tokenizer.decode(part, skip_special_tokens=False, clean_up_tokenization_spaces=False))

    return {"text": chunks}


# Load the hyperswitch code dataset
ds = load_dataset("archit11/hyperswitch-code-dataset")
train_ds = ds["train"] if isinstance(ds, dict) and "train" in ds else ds
train_ds = train_ds.map(formatting_examples, batched=True, remove_columns=train_ds.column_names)
train_ds = train_ds.map(chunk_long_texts, batched=True, num_proc=1)


# Training config — same overall settings, adjusted for ZeRO-3
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
    output_dir="outputs-hyperswitch-lora",
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
print(f"  Warmup ratio: {warmup_ratio}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=training_args,
)

trainer.train()

model.save_pretrained("hyperswitch_lora")
tokenizer.save_pretrained("hyperswitch_lora")
