# coding=utf-8
import argparse
import json
import os
import shutil
from tqdm import tqdm

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def _resolve_runtime(args):
    """
    Decide whether to run on CPU or CUDA and pick safe precision/dtype defaults.
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")

    use_cpu = (args.device == "cpu") or (args.device == "auto" and not torch.cuda.is_available())

    # Mixed precision: safest default is fp32 on CPU
    mixed_precision = args.mixed_precision
    if use_cpu:
        mixed_precision = "no"

    # Choose model dtype consistent with mixed precision (but force fp32 on CPU)
    if use_cpu:
        torch_dtype = torch.float32
    else:
        if mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    # Attention implementation: sdpa is OK on many setups, but eager is safest on CPU
    attn_impl = args.attn_implementation
    if attn_impl is None:
        attn_impl = "eager" if use_cpu else "sdpa"

    return use_cpu, mixed_precision, torch_dtype, attn_impl


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

    # NEW: device/precision controls
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto: use CUDA if available else CPU; cpu: force CPU; cuda: force CUDA")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16",
                        help="Ignored on CPU (forced to 'no'). On CUDA, choose no/fp16/bf16.")
    parser.add_argument("--attn_implementation", choices=["sdpa", "eager"], default=None,
                        help="If unset: defaults to eager on CPU, sdpa on CUDA.")

    args = parser.parse_args()

    use_cpu, mixed_precision, torch_dtype, attn_impl = _resolve_runtime(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        cpu=use_cpu,
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl, "r", encoding="utf-8").readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        # pin_memory only helps for CUDA; harmless on CPU but you can keep it False
        pin_memory=(accelerator.device.type == "cuda"),
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
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
                param_dtype = next(model.parameters()).dtype  # safer than model.dtype

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(device=device, dtype=param_dtype)
                ).detach()

                if target_speaker_embedding is None:
                    # Keep a CPU copy for saving later (works for both CPU/GPU training)
                    target_speaker_embedding = speaker_embedding.detach().to("cpu")

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # text
                input_text_embedding = model.talker.model.text_embedding(input_text_ids)

                # IMPORTANT: 0.6B needs this projection (dims 2048 -> 1024)
                if hasattr(model.talker, "text_projection") and model.talker.text_projection is not None:
                    input_text_embedding = model.talker.text_projection(input_text_embedding)

                input_text_embedding = input_text_embedding * text_embedding_mask

                # codec
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                # debug (optional)
                # accelerator.print("text emb", input_text_embedding.shape, "codec emb", input_codec_embedding.shape)

                input_embeddings = input_text_embedding + input_codec_embedding


                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
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

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"[device={accelerator.device.type} mp={mixed_precision}] "
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            for k in [k for k in state_dict.keys() if k.startswith(drop_prefix)]:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()
