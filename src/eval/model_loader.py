"""
Model loading for evaluation — HuggingFace and vLLM backends.
"""
import gc
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_config):
    """Load model + tokenizer. Handles base model, LoRA, and full FT checkpoints."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = model_config["checkpoint"]
    is_lora = model_config.get("lora", False)
    is_full_ft = model_config.get("full_ft", False)

    if not is_lora and not is_full_ft and not model_config.get("is_baseline", False):
        meta_path = Path(checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"Auto-detected from training_meta.json: lora={is_lora}, full_ft={is_full_ft}")

    if is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        logger.info(f"Loading LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    elif is_full_ft:
        logger.info(f"Loading full FT checkpoint: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    else:
        logger.info(f"Loading base model: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def load_mm_model(model_config):
    """Load Memory Manager model for use_mm inference. Returns (model, tokenizer) or (None, None)."""
    mm_checkpoint = model_config.get("mm_checkpoint")
    if not mm_checkpoint:
        logger.warning("use_mm=true but no mm_checkpoint specified, falling back to heuristic")
        return None, None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_full_ft = model_config.get("full_ft", False)
    is_lora = model_config.get("lora", False)

    if not is_lora and not is_full_ft:
        meta_path = Path(mm_checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"MM auto-detected: lora={is_lora}, full_ft={is_full_ft}")

    if is_full_ft:
        logger.info(f"Loading MM full FT checkpoint: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    elif is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        mm_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_model = PeftModel.from_pretrained(mm_model, mm_checkpoint)
        mm_model = mm_model.merge_and_unload()
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    else:
        logger.info(f"Loading MM base model: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)

    if mm_tokenizer.pad_token is None:
        mm_tokenizer.pad_token = mm_tokenizer.eos_token
    mm_model.eval()
    return mm_model, mm_tokenizer


def _detect_checkpoint_type(checkpoint, model_config):
    """Auto-detect LoRA vs full FT from training_meta.json. Returns (is_lora, is_full_ft)."""
    is_lora = model_config.get("lora", False)
    is_full_ft = model_config.get("full_ft", False)
    if not is_lora and not is_full_ft and not model_config.get("is_baseline", False):
        meta_path = Path(checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"Auto-detected: lora={is_lora}, full_ft={is_full_ft}")
    return is_lora, is_full_ft


def _merge_lora_to_tmpdir(checkpoint, base_model_name):
    """Merge LoRA adapter into base model and save to a temp directory for vLLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA {checkpoint} into {base_model_name} for vLLM...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, checkpoint)
    merged = merged.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tmpdir = tempfile.mkdtemp(prefix="vllm_merged_")
    merged.save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)
    del base, merged
    gc.collect()
    logger.info(f"Merged LoRA checkpoint saved to {tmpdir}")
    return tmpdir


def load_model_vllm(model_config, gpu_memory_utilization=0.85,
                     tensor_parallel_size=1, max_model_len=4096):
    """Load model via vLLM offline LLM engine. Returns vllm.LLM instance."""
    from vllm import LLM

    checkpoint = model_config["checkpoint"]
    is_lora, is_full_ft = _detect_checkpoint_type(checkpoint, model_config)

    if is_lora:
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        model_path = _merge_lora_to_tmpdir(checkpoint, base_model_name)
    else:
        model_path = checkpoint

    logger.info(f"Loading vLLM model: {model_path} (tp={tensor_parallel_size}, "
                f"gpu_mem={gpu_memory_utilization})")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return llm


def load_mm_model_vllm(model_config, gpu_memory_utilization=0.45,
                        tensor_parallel_size=1, max_model_len=4096):
    """Load Memory Manager model via vLLM. Returns vllm.LLM instance or None."""
    from vllm import LLM

    mm_checkpoint = model_config.get("mm_checkpoint")
    if not mm_checkpoint:
        logger.warning("use_mm=true but no mm_checkpoint specified, falling back to heuristic")
        return None

    is_lora, is_full_ft = _detect_checkpoint_type(mm_checkpoint, model_config)

    if is_lora:
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        model_path = _merge_lora_to_tmpdir(mm_checkpoint, base_model_name)
    else:
        model_path = mm_checkpoint

    logger.info(f"Loading vLLM MM model: {model_path} (tp={tensor_parallel_size}, "
                f"gpu_mem={gpu_memory_utilization})")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return llm
