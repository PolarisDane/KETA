import json
from os import path as osp
import os
from copy import deepcopy
from tqdm import tqdm

idx = 0

dataset_dir = r"HumanML3D/HumanML3D"
if not osp.exists(dataset_dir):
    raise ValueError("Dataset not exist!")

md = osp.join(dataset_dir, "all.txt")
with open(md, "r") as md_reader:
    files = md_reader.readlines()

fds = {}
batch_files = {}

def get_fd(rm=False, mode="a"):
    global idx
    i = idx // 4000
    idx += 1
    if i in fds:
        return f"cv/batch/batch_file_{i}.jsonl", fds[i], (idx-1) % 4000
    if osp.exists(f"cv/batch/batch_file_{i}.jsonl") and rm:
        os.remove(f"cv/batch/batch_file_{i}.jsonl")
    batch_files[f"cv/batch/batch_file_{i}.jsonl"] = {}
    fds[i] = open(f"cv/batch/batch_file_{i}.jsonl", mode)
    return f"cv/batch/batch_file_{i}.jsonl", fds[i], (idx-1) % 4000

def llama():
    import transformers
    import numpy as np
    local_model_path = "/aidata/qiaojun/jy_cv/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

    model = transformers.AutoModel.from_pretrained(local_model_path, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)

    pipeline = transformers.pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    
    for f in tqdm(files[:]):
        with open(osp.join(osp.join(dataset_dir, "texts"), f.strip()+".txt"), "r") as fr:
            texts = fr.readlines()
        npz_files = dict(np.load(osp.join("cv/npz", f.strip()+".npz")))
        max_len = npz_files['spt_encoding'].shape[1]
        with open(osp.join("cv/text_p", f.strip()+".jsonl"), "r") as fr:
            text_p = fr.readlines()
        text_p_idx = 0
        npz_idx = 0
        len_dict = []
        ori_encoding = []
        spt_encoding = []
        f_tags = []
        e_tags = []
        
        for text in texts:
            text = text.strip()
            if "[" in text:
                text = text.split("]")[-1]
            text, _, f_tag, e_tag = text.split("#")
            text = text.strip().rstrip(".") + '.'
            f_tag = float(f_tag)
            e_tag = float(e_tag)
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            e_tag = 0.0 if np.isnan(e_tag) else e_tag
            if text_p_idx < len(text_p):
                text_p_p = json.loads(text_p[text_p_idx].strip())
                cp_text = text_p_p['origin']
            else:
                cp_text = ""
            if text == cp_text:
                len_dict.append(npz_files['len'][npz_idx])
                ori_encoding.append(npz_files['ori_encoding'][npz_idx])
                spt_encoding.append(npz_files['spt_encoding'][npz_idx])
                f_tags.append(f_tag)
                e_tags.append(e_tag)
                text_p_idx += 1
                npz_idx += 1
            else:
                len_dict.append(0)
                output = pipeline([text])
                origin_output = np.array(output[0][0])
                origin_output = np.mean(origin_output, axis=0)
                ori_encoding.append(origin_output)
                gpt_output = np.zeros((max_len, len(origin_output)))
                gpt_output[0] = origin_output
                spt_encoding.append(gpt_output)
                f_tags.append(f_tag)
                e_tags.append(e_tag)
            
        
        assert text_p_idx == len(text_p), f"{f.strip()} {text_p_idx} {len(text_p)}"
        np.savez(osp.join("cv/npzz", f.strip()), 
                    #  origin_text=line['origin'],
                    #  gpt_text=line['gpt'],
                        len=len_dict,
                    #  origin_encoding_token=output[0],
                        ori_encoding=ori_encoding,
                    #  gpt_encoding_token=output[1:],
                        spt_encoding=spt_encoding,
                        f_tags=f_tags,
                        e_tags=e_tags)
                    
        
    
if __name__ == "__main__":
    llama()
            