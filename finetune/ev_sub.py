import json, sys, torch
from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig
from dataset import TTSDataset
from sft_12hz import sub_talker_loss_aligned

BASE = sys.argv[1]
rows=[json.loads(l) for l in open("train_with_codes.jsonl",encoding="utf-8").readlines()[:8]]
base_tts = Qwen3TTSModel.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
SPK = base_tts.model.speaker_encoder.eval()

def ev(path, tag):
    m=Qwen3TTSModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model=m.model.eval(); cfg=AutoConfig.from_pretrained(path)
    ds=TTSDataset(rows,m.processor,cfg); bt=ds.collate_fn([ds[i] for i in range(len(rows))])
    dev,dt=model.device,model.dtype
    ids=bt["input_ids"].to(dev); cid=bt["codec_ids"].to(dev)
    tm=bt["text_embedding_mask"].to(dev).to(dt); cme=bt["codec_embedding_mask"].to(dev).to(dt)
    at=bt["attention_mask"].to(dev); lb=bt["codec_0_labels"].to(dev)
    cm=bt["codec_mask"].to(dev); rm=bt["ref_mels"].to(dev).to(dt)
    with torch.no_grad():
        spk=SPK(rm)
        te=model.talker.text_projection(model.talker.get_text_embeddings()(ids[:,:,0]))*tm
        ce=model.talker.model.codec_embedding(ids[:,:,1])*cme; ce[:,6,:]=spk
        emb=te+ce
        for i in range(1,16):
            emb=emb+model.talker.code_predictor.get_input_embeddings()[i-1](cid[:,:,i])*cm.unsqueeze(-1).to(dt)
        o=model.talker(inputs_embeds=emb[:,:-1,:],attention_mask=at[:,:-1],labels=lb[:,1:],output_hidden_states=True)
        hs=o.hidden_states[0][-1]
        lg,ls=sub_talker_loss_aligned(model.talker, cid[cm], hs[cm[:,:-1]])
    tgt=cid[cm][:,1:]
    accs=[(lg.argmax(-1)[:,j]==tgt[:,j]).float().mean().item() for j in range(15)]
    print(f"[{tag}] main {o.loss.item():.4f}  sub {ls.item():.3f}  mean_acc {sum(accs)/15:.4f}")
    print(f"[{tag}] per-cb: {' '.join(f'{a:.2f}' for a in accs)}")
    del m; torch.cuda.empty_cache()

ev(BASE, "base")
ev("output/checkpoint-epoch-7", "ep7")
ev("output/checkpoint-epoch-23", "ep23")
ev("output_overfit/checkpoint-epoch-18", "overfit")
