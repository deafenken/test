"""Smoke test: load Qwen2.5-VL-7B from LOCAL snapshot, eager attention, per-head visual mass."""
import os, torch, numpy as np
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

SNAP="/home/intern/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
dev="cuda:0"  # physical GPU set via CUDA_VISIBLE_DEVICES
print("loading from local snapshot...", flush=True)
proc=AutoProcessor.from_pretrained(SNAP, local_files_only=True)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    SNAP, torch_dtype=torch.bfloat16, attn_implementation="eager",
    device_map={"":dev}, local_files_only=True)
model.eval(); print("loaded bf16 eager OK", flush=True)

img=Image.fromarray((np.random.rand(256,256,3)*255).astype("uint8"))
msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"What color dominates? One word."}]}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
imgs,vids=process_vision_info(msgs)
inputs=proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(dev)

itid=model.config.image_token_id
ids=inputs["input_ids"][0]; img_mask=(ids==itid); n_img=int(img_mask.sum())
print(f"seq_len={ids.shape[0]} n_image_tokens={n_img} image_token_id={itid}", flush=True)
with torch.no_grad():
    out=model(**inputs, output_attentions=True, use_cache=False)
att=out.attentions; L=len(att); H=att[0].shape[1]
print(f"n_layers={L} heads/layer={H} total_heads={L*H} attn0.shape={tuple(att[0].shape)}", flush=True)
li=L//2
vm=att[li][0,:,-1,:][:,img_mask].sum(-1)
print(f"layer{li} per-head visual mass min/mean/max={vm.min():.3f}/{vm.mean():.3f}/{vm.max():.3f}", flush=True)
# also generate 1 answer to confirm end-to-end
gen=model.generate(**inputs, max_new_tokens=5, do_sample=False)
print("gen:", proc.batch_decode(gen[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip(), flush=True)
print("SMOKE_OK", flush=True)
