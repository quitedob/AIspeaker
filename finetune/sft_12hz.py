# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def sub_talker_loss_aligned(talker, codec_ids, hidden_states):
    """子 talker 前向 + 位置对齐的交叉熵。

    官方 `forward_sub_talker_finetune` 内部走 `ForCausalLMLoss`，那个函数会再做一次
    `shift_labels`；但这里 logits 已按 codebook 组逐位置对齐、labels 也已切成
    `codec_ids[:, 1:]`，再 shift 就会错位一格。实测在 base 模型上：
    双重 shift loss 12.72 / acc 0.02%，正确对齐 loss 5.68 / acc 17%。
    错位的子 talker 在推理时会产出乱码 codebook 并反馈进主 talker，
    导致 EOS 永远无法触发（生成一直顶到 max_new_tokens）。
    """
    sub_embeds = [hidden_states.unsqueeze(1)]
    for i in range(talker.config.num_code_groups - 1):
        emb_layer = (
            talker.get_input_embeddings() if i == 0
            else talker.code_predictor.get_input_embeddings()[i - 1]
        )
        sub_embeds.append(emb_layer(codec_ids[:, i:i + 1]))
    sub_embeds = torch.cat(sub_embeds, dim=1)

    out = talker.code_predictor.forward_finetune(
        inputs_embeds=sub_embeds, labels=codec_ids[:, 1:]
    )
    logits = out.logits
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        codec_ids[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    return logits, loss

target_speaker_embedding = None
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
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="Attention backend. sdpa needs no flash-attn install.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (batch_size * grad_accum = effective batch).",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to cut activation memory.",
    )
    parser.add_argument(
        "--no_grad_checkpointing",
        dest="grad_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing (more VRAM, faster).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write a checkpoint every N epochs (0 = never). Each is ~2.4GB.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (-1 = run all epochs).",
    )
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=args.attn,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # 16GB 卡必需：不存全部中间激活，反向时重算
    if args.grad_checkpointing:
        qwen3tts.model.gradient_checkpointing_enable()
        qwen3tts.model.config.use_cache = False
        accelerator.print("[mem] gradient checkpointing enabled")

    train_data = open(args.train_jsonl, encoding="utf-8").readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    def save_checkpoint(epoch):
        if not accelerator.is_main_process:
            return
        output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

        input_config_file = os.path.join(MODEL_PATH, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {
            args.speaker_name: 3000
        }
        talker_config["spk_is_dialect"] = {
            args.speaker_name: False
        }
        config_dict["talker_config"] = talker_config

        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

        drop_prefix = "speaker_encoder"
        keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
        for k in keys_to_drop:
            del state_dict[k]

        weight = state_dict['talker.model.codec_embedding.weight']
        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        accelerator.print(f"[save] {output_dir}")

    global_step = 0
    stop = False

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # 0.6B 的 text_hidden_size(2048) != hidden_size(1024)，
                # 必须过 text_projection；1.7B 里它是恒等的，官方脚本因此省略了
                input_text_embedding = model.talker.text_projection(
                    model.talker.get_text_embeddings()(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = sub_talker_loss_aligned(
                    model.talker, talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    global_step += 1

            if step % 10 == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.max_memory_reserved() / 1024**3
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | OptStep {global_step} | Loss: {loss.item():.4f} | "
                    f"main {outputs.loss.item():.4f} sub {sub_talker_loss.item():.4f} | "
                    f"peak {peak:.2f}GB reserved {reserved:.2f}GB"
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(epoch)

        if stop:
            if not (args.save_every and (epoch + 1) % args.save_every == 0):
                save_checkpoint(epoch)
            accelerator.print(f"[done] stopped at opt step {global_step}")
            break

if __name__ == "__main__":
    train()
