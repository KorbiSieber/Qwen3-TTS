# coding=utf-8
"""
Qwen3-TTS 12Hz SFT script (reworked) that:
- Works with HF model IDs (e.g. "Qwen/Qwen3-TTS-12Hz-0.6B-Base") by resolving to a local snapshot dir.
- Fixes the 0.6B embedding-dim mismatch by applying talker.text_projection (when present).
- Supports CPU/CUDA + mixed precision controls.
- Saves checkpoints by copying the resolved local model dir + writing updated config + writing model.safetensors.
- NEW: After each epoch, generates a TTS audio sample and saves it as a WAV.

Usage example:
  uv run python sft_12hz_reworked.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output_model_path runs/obama/qwen3tts_output \
    --train_jsonl ../../runs/obama/qwen3tts_dataset/train_with_codes.jsonl \
    --batch_size 1 --lr 2e-6 --num_epochs 2 --speaker_name obama --device cpu
"""

import argparse
import json
import os
import shutil
from typing import Tuple, Optional

import torch
import torchaudio
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from huggingface_hub import snapshot_download

target_speaker_embedding = None


def _resolve_runtime(args) -> Tuple[bool, str, torch.dtype, str]:
    """
    Decide whether to run on CPU or CUDA and pick safe precision/dtype defaults.
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")

    use_cpu = (args.device == "cpu") or (args.device == "auto" and not torch.cuda.is_available())

    mixed_precision = args.mixed_precision
    if use_cpu:
        mixed_precision = "no"

    if use_cpu:
        torch_dtype = torch.float32
    else:
        if mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    attn_impl = args.attn_implementation
    if attn_impl is None:
        attn_impl = "eager" if use_cpu else "sdpa"

    return use_cpu, mixed_precision, torch_dtype, attn_impl


def resolve_model_dir(model_id_or_path: str) -> str:
    """
    If model_id_or_path is a local directory, return it.
    Otherwise treat it as a Hugging Face repo id and download/resolve the local snapshot dir.
    """
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return snapshot_download(repo_id=model_id_or_path)


def load_qwen3tts(
    model_dir: str,
    torch_dtype: torch.dtype,
    attn_impl: str,
    device_map: Optional[str] = None,
) -> Qwen3TTSModel:
    """
    Compatibility wrapper:
    - Some builds use dtype=..., some use torch_dtype=...
    - Some accept device_map=..., some don't
    We try a few combinations.
    """
    attempts = []

    # Prefer dtype + device_map
    attempts.append(dict(dtype=torch_dtype, attn_implementation=attn_impl, device_map=device_map))
    # Fallback: torch_dtype + device_map
    attempts.append(dict(torch_dtype=torch_dtype, attn_implementation=attn_impl, device_map=device_map))
    # Fallback: dtype without device_map
    attempts.append(dict(dtype=torch_dtype, attn_implementation=attn_impl))
    # Fallback: torch_dtype without device_map
    attempts.append(dict(torch_dtype=torch_dtype, attn_implementation=attn_impl))

    last_err = None
    for kwargs in attempts:
        try:
            # Remove device_map if None
            if kwargs.get("device_map", "sentinel") is None:
                kwargs.pop("device_map", None)
            return Qwen3TTSModel.from_pretrained(model_dir, **kwargs)
        except TypeError as e:
            last_err = e
            continue

    raise last_err if last_err is not None else RuntimeError("Failed to load Qwen3TTSModel")


@torch.inference_mode()
def generate_and_save_sample(
    ckpt_dir: str,
    speaker_name: str,
    text: str,
    out_wav_path: str,
    device_map: Optional[str],
    torch_dtype: torch.dtype,
    attn_impl: str,
    accelerator: Accelerator,
):
    """
    Load checkpoint as a fresh Qwen3TTSModel wrapper and generate a single WAV sample.
    Runs only on main process.
    """
    try:
        tts = load_qwen3tts(
            ckpt_dir,
            torch_dtype=torch_dtype,
            attn_impl=attn_impl,
            device_map=device_map,
        )

        wavs, sr = tts.generate_custom_voice(
            text=text,
            speaker=speaker_name,
            max_duration=10.0,
        )

        wav0 = wavs[0]
        if isinstance(wav0, torch.Tensor):
            wav_t = wav0
        else:
            # likely numpy
            wav_t = torch.from_numpy(wav0)

        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)  # [1, T]
        elif wav_t.dim() == 2 and wav_t.shape[0] != 1:
            # if [T, C] or multi-channel, force mono
            if wav_t.shape[1] == 1:
                wav_t = wav_t.transpose(0, 1)
            else:
                wav_t = wav_t.mean(dim=0, keepdim=True)

        wav_t = wav_t.to(torch.float32).clamp(-1.0, 1.0).cpu()

        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        torchaudio.save(out_wav_path, wav_t, sr, encoding="PCM_S", bits_per_sample=16)
        accelerator.print(f"Saved sample WAV: {out_wav_path}")
    except Exception as e:
        accelerator.print(f"[WARN] Sample generation failed at {ckpt_dir}: {repr(e)}")


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")

    # Training knobs
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # device/precision controls
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto: use CUDA if available else CPU; cpu: force CPU; cuda: force CUDA",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Ignored on CPU (forced to 'no'). On CUDA, choose no/fp16/bf16.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "eager"],
        default=None,
        help="If unset: defaults to eager on CPU, sdpa on CUDA.",
    )

    # Logging
    parser.add_argument(
        "--log_with",
        choices=["none", "tensorboard"],
        default="none",
        help="If tensorboard, you need tensorboard installed; otherwise use none.",
    )

    # NEW: sample generation
    parser.add_argument(
        "--no_epoch_sample",
        action="store_true",
        help="Disable generating a sample WAV after each epoch.",
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default="Hello! This is a fine-tuning checkpoint sample.",
        help="Text to synthesize after each epoch.",
    )
    parser.add_argument(
        "--sample_dirname",
        type=str,
        default="samples",
        help="Subdirectory name inside each checkpoint folder to write sample wavs.",
    )
    parser.add_argument(
        "--sample_filename",
        type=str,
        default="sample.wav",
        help="Filename for the generated WAV inside the checkpoint sample directory.",
    )

    args = parser.parse_args()

    use_cpu, mixed_precision, torch_dtype, attn_impl = _resolve_runtime(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=(None if args.log_with == "none" else args.log_with),
        cpu=use_cpu,
    )

    # Resolve HF id -> local snapshot dir
    MODEL_ID = args.init_model_path
    MODEL_DIR = resolve_model_dir(MODEL_ID)

    # Load model + config from local dir
    qwen3tts = load_qwen3tts(MODEL_DIR, torch_dtype=torch_dtype, attn_impl=attn_impl, device_map=None)
    config = AutoConfig.from_pretrained(MODEL_DIR)

    # Load dataset
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()][:100]

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)
    model.train()

    for epoch in range(args.num_epochs):
        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                device = accelerator.device
                param_dtype = next(model.parameters()).dtype

                # Speaker embedding from ref mels
                speaker_embedding = model.speaker_encoder(ref_mels.to(device=device, dtype=param_dtype)).detach()

                # Store the first batch speaker embedding for later checkpoint injection (CPU copy)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding.detach().to("cpu")

                # Split text + codec ids (packed in input_ids)
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # ---- Build input embeddings (fix for 0.6B: apply text_projection if present) ----
                input_text_embedding = model.talker.model.text_embedding(input_text_ids)

                # IMPORTANT: 0.6B needs this projection (e.g., 2048 -> 1024)
                if hasattr(model.talker, "text_projection") and model.talker.text_projection is not None:
                    input_text_embedding = model.talker.text_projection(input_text_embedding)

                input_text_embedding = input_text_embedding * text_embedding_mask

                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                # Inject speaker embedding at position 6 (as per original script)
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                # Add additional codec embeddings
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "device": accelerator.device.type, "mp": mixed_precision})

            if step % 10 == 0:
                accelerator.print(
                    f"[device={accelerator.device.type} mp={mixed_precision}] "
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        # ---------------------------
        # Save checkpoint (main proc)
        # ---------------------------
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            # Copy base model files from resolved local snapshot dir
            shutil.copytree(MODEL_DIR, output_dir, dirs_exist_ok=True)

            # Patch config.json for custom voice speaker mapping
            input_config_file = os.path.join(MODEL_DIR, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")

            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {}) or {}

            spk_id = talker_config.get("spk_id", {}) or {}
            spk_is_dialect = talker_config.get("spk_is_dialect", {}) or {}

            spk_id[args.speaker_name] = 3000
            spk_is_dialect[args.speaker_name] = False

            talker_config["spk_id"] = spk_id
            talker_config["spk_is_dialect"] = spk_is_dialect
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Save model weights (drop speaker_encoder, inject speaker embedding row)
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # Drop speaker encoder weights
            drop_prefix = "speaker_encoder"
            for k in [k for k in list(state_dict.keys()) if k.startswith(drop_prefix)]:
                del state_dict[k]

            # Inject speaker embedding into codec embedding table row 3000
            weight_key = "talker.model.codec_embedding.weight"
            if weight_key not in state_dict:
                raise KeyError(f"Expected '{weight_key}' in state_dict but it was not found.")

            weight = state_dict[weight_key]
            if weight.shape[0] <= 3000:
                raise RuntimeError(
                    f"codec_embedding.weight has only {weight.shape[0]} rows; cannot write row 3000."
                )

            if target_speaker_embedding is None:
                raise RuntimeError("target_speaker_embedding is None; did training run any steps?")

            emb_row = target_speaker_embedding[0].detach().to(weight.dtype)
            state_dict[weight_key][3000] = emb_row

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

            accelerator.print(f"Saved checkpoint: {output_dir}")

            if not args.no_epoch_sample:
                # device_map for inference
                # - If training on CUDA, try "cuda:0" (most common)
                # - If CPU, leave None
                infer_device_map = "cuda:0" if accelerator.device.type == "cuda" else None

                sample_dir = os.path.join(output_dir, args.sample_dirname)
                sample_path = os.path.join(sample_dir, args.sample_filename)

                generate_and_save_sample(
                    ckpt_dir=output_dir,
                    speaker_name=args.speaker_name,
                    text=args.sample_text,
                    out_wav_path=sample_path,
                    device_map=infer_device_map,
                    torch_dtype=torch_dtype,
                    attn_impl=attn_impl,
                    accelerator=accelerator,
                )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
