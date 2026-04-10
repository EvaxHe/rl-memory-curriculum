"""
Model loading for GRPO training via Unsloth.
"""
import logging

logger = logging.getLogger(__name__)


def load_model_unsloth(config: dict, max_seq_length: int = 2048):
    """Load model and tokenizer via Unsloth's FastLanguageModel.

    Handles both LoRA and full fine-tuning.
    Qwen3.5 is not yet supported by vLLM, so fast_inference is disabled.
    """
    from unsloth import FastLanguageModel

    model_name = config["model"]["name"]
    use_lora = config["training"].get("use_lora", True)
    seed = config["experiment"].get("seed", 42)

    logger.info(f"Loading model via Unsloth: {model_name}")
    logger.info(f"Mode: {'LoRA' if use_lora else 'Full fine-tuning'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        fast_inference=False,  # Qwen3.5 not supported by vLLM yet
        full_finetuning=not use_lora,
    )

    if use_lora:
        lora_r = config["training"].get("lora_rank", 16)
        lora_alpha = config["training"].get("lora_alpha", 16)
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            max_seq_length=max_seq_length,
        )

    return model, tokenizer
