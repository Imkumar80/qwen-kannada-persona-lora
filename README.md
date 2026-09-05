# Qwen2.5-0.5B Kannada Persona LoRA

A small hands-on fine-tuning experiment to teach **Qwen/Qwen2.5-0.5B-Instruct** a conversational Kannada persona named **Kavi**.

## Goal

This project explores whether a very small LoRA/SFT fine-tuning run can improve Kannada conversational behavior and establish a consistent persona without updating the full model.

## Base model

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Parameters: ~0.5B
- Fine-tuning: LoRA + Supervised Fine-Tuning
- Hardware: Google Colab Tesla T4

## Persona

Kavi is designed to:

- Respond naturally in conversational Kannada
- Use casual Kannada rather than textbook/formal Kannada
- Mix Kannada with common English words naturally
- Give friendly, relatively short responses
- Maintain a consistent fictional personality
- Occasionally use light humor/emojis

## Training setup

The first experiment used 20 conversational examples for 10 epochs.

LoRA configuration:

- `r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Learning rate: `2e-4`
- Batch size: 2
- Gradient accumulation: 4
- Maximum sequence length: 512

Only about **0.44% of the model parameters** were trainable (~2.16M LoRA parameters out of ~496M total parameters).

## Training result

The first run completed successfully:

- Epochs: 10
- Steps: 30
- Final reported training loss: `1.8767` at the final step
- Mean training loss reported by Trainer: `2.75285`

Because the initial dataset is very small, these results should be treated as an experiment rather than evidence of broad Kannada capability. Evaluation on unseen prompts is important to distinguish generalization from memorization.

## Project structure

```text
qwen-kannada-persona-lora/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── kavi_persona_dataset.jsonl
├── notebooks/
│   └── kavi_qwen_lora.ipynb
└── src/
    ├── train.py
    └── inference.py
```

## Reproduce

Install dependencies:

```bash
pip install -U transformers datasets peft trl accelerate bitsandbytes torchao
```

Then run the training notebook/script on a CUDA GPU.

## Next experiments

1. Compare baseline vs fine-tuned responses on the same 10 prompts.
2. Test on unseen Kannada prompts.
3. Add multi-turn conversations.
4. Expand the dataset from 20 examples to a larger, more diverse set.
5. Evaluate persona consistency and Kannada naturalness separately.

## Note

The base Qwen checkpoint is not included in this repository. It is downloaded from Hugging Face when the code runs.
