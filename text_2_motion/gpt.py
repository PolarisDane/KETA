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

def process_gpt_output(line, response):
    if response.count('[') == 0:
        return False, None
    response = response.split("[")[-1].split("]")[0]
    responses = response.split(",")
    ret = []
    for response in responses:
        response = response.strip().lstrip("\"").rstrip("\"")
        if len(response) == 0:
            continue
        ret.append(response)
    
    if len(ret) == 0: # failed
        ret = [line]
    return True, ret


    


def create():
    module = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "system",
            "content": "You are a highly specialized assistant designed to analyze and process textual descriptions of human postures and actions. Your primary function is to decompose these descriptions into fine-grained actions arranged chronologically. Focus on detecting and interpreting sequence markers like 'then,' 'twice,' 'again,' and other words indicating repetitions or transitions. Ensure that your decomposition explicitly outlines: \n1. The initial state of the posture or action. \n2. Detailed intermediate steps. \n3. The final state. \nAll outputs must be formatted as a single-layer Python list, free from nested structures, for ease of integration with other tools. Validate the syntax of the generated Python list to ensure correctness, such as properly matched brackets, quotes, and commas. If issues are detected, correct them before presenting the output. Accuracy, clarity, and attention to chronological details are critical in your responses."
            },
            {
            "role": "user",
            "content": "a man kicks something or someone with his left leg."
            }
        ],
        "max_tokens": 1000
        }
    }

    for f in files:
        f = f.strip()
        text_file = osp.join(osp.join(dataset_dir, "texts"), f+".txt")
        with open(text_file, "r") as text_reader:
            lines = text_reader.readlines()
        for line in lines:
            line = line.strip()
            if "[" in line:
                line = line.split("]")[-1]
            line = line.split("#")[0]
            line = line.strip().rstrip(".") + '.'
            _, fd, _ = get_fd(rm=True)
            module['custom_id'] = f"request-{idx}"
            module['body']['messages'][1]['content'] = line
            fd.write(json.dumps(module) + '\n')

    for f in batch_files:
        if osp.exists(f):
            batch_input_file = client.files.create(
            file=open(f, "rb"),
            purpose="batch"
            )
            batch_files[f]["file_id"]=batch_input_file.id
        else:
            raise ValueError('Not Existing Batch File')

    for batch_file in batch_files:
        batch_input_file_id = batch_files[batch_file]['file_id']
        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_files[batch_file]['output_file'] = batch_obj.output_file_id
        batch_files[batch_file]['batch_id'] = batch_obj.id

    with open("cv/gpt.jsonl", "w") as fw:
        json.dump(batch_files, fw)

def retrieve():
    with open("cv/gpt.json", "r") as fi:
        gpts = json.load(fi)
    for id in gpts:
        batch = client.batches.retrieve(gpts[id]['batch_id'])
        print(id, batch.status, batch.request_counts)
        gpts[id]['output_file'] = batch.output_file_id
    with open("cv/gpt.json", "w") as fw:
        json.dump(gpts, fw)

def cancel():
    lists = client.batches.list(limit=100).data
    for l in lists:
        client.batches.cancel(l.id)

def integrate():
    with open("cv/gpt.json", "r") as fi:
        gpts = json.load(fi)
    for batch_file in gpts:
        output_id = gpts[batch_file]["output_file"]
        if osp.exists(f"cv/batch/{output_id}.jsonl"):
            pass
        else:
            file_response = client.files.content(output_id)
            with open(f"cv/batch/{output_id}.jsonl", "w") as test:
                test.write(file_response.text)

    missing_value = []
    missing_batch_value = []
    batch_md = {}
    retrieve_data = {}
    if "cv/batch/missing_batch.jsonl" in gpts:
        output_id = gpts["cv/batch/missing_batch.jsonl"]["output_file"]
        with open(f"cv/batch/{output_id}.jsonl", "r") as output_reader:
            for line in output_reader:
                missing_batch_value.append(json.loads(line)['custom_id'])
                retrieve_data[json.loads(line)['custom_id']] = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]

    for batch_file in gpts:
        
        batch_to_process, batch_output = [], []
        with open(batch_file, "r") as batch_reader:
            for line in batch_reader:
                batch_to_process.append(json.loads(line))
        output_id = gpts[batch_file]["output_file"]
        with open(f"cv/batch/{output_id}.jsonl", "r") as output_reader:
            for line in output_reader:
                batch_output.append(json.loads(line)['custom_id'])
                retrieve_data[json.loads(line)['custom_id']] = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
        for data in batch_to_process:
            if data['custom_id'] not in batch_output and data['custom_id'] not in missing_batch_value:
                missing_value.append(data)
        batch_md[batch_file] = batch_to_process

    print(len(missing_value))
    # if len(missing_value) != 0:
    #     with open("cv/missing_batch.jsonl", "w") as fo:
    #         for data in missing_value:
    #             fo.write(json.dumps(data) + '\n')
    #     batch_input_file = client.files.create(
    #         file=open("cv/missing_batch.jsonl", "rb"),
    #         purpose="batch"
    #         )
    #     gpts["cv/missing_batch.jsonl"] = {}
    #     gpts["cv/missing_batch.jsonl"]["file_id"]=batch_input_file.id
    #     batch_obj = client.batches.create(
    #         input_file_id=batch_input_file.id,
    #         endpoint="/v1/chat/completions",
    #         completion_window="24h",
    #     )
    #     gpts["cv/missing_batch.jsonl"]['output_file'] = batch_obj.output_file_id
    #     gpts["cv/missing_batch.jsonl"]['batch_id'] = batch_obj.id
    #     with open("cv/gpt.json", "w") as fw:
    #         json.dump(gpts, fw)         
    assert len(missing_value) == 0, "There are some missing data to process"

    for f in tqdm(files):
        file_flag = False
        f = f.strip()
        text_file = osp.join(osp.join(dataset_dir, "texts"), f+".txt")
        with open(text_file, "r") as text_reader:
            lines = text_reader.readlines()
        for line in lines:
            line = line.strip()
            if "[" in line:
                line = line.split("]")[-1]
            line = line.split("#")[0]
            line = line.strip().rstrip(".") + '.'

            batch_file_name, _, batch_idx = get_fd(mode="r")
            assert batch_md[batch_file_name][batch_idx]["body"]["messages"][1]["content"] == line, f"{line}  {batch_file_name} {batch_idx}"

            response = retrieve_data[batch_md[batch_file_name][batch_idx]['custom_id']]
            line_flag, line_ret = process_gpt_output(line, response)
            file_flag = file_flag | line_flag
            if line_flag:
                assert len(line_ret) != 0, f"{line} {response}"
                with open(osp.join("cv/text_p", f+'.jsonl'), "a") as fw:
                    fw.write(json.dumps({"origin": line, "gpt": line_ret}) + '\n')

        if not file_flag:
            print(f)

    # print(two_len)

def llama_encoding():
    import transformers
    import numpy as np
    local_model_path = "/aidata/qiaojun/jy_cv/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

    model = transformers.AutoModel.from_pretrained(local_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)

    pipeline = transformers.pipeline("feature-extraction", model=model, tokenizer=tokenizer, device="cuda:2")
    
    for f in tqdm(files):
        max_len = 0
        with open(osp.join("cv/text_p", f.strip()+'.jsonl'), "r") as fr:
            for line in fr:
                line = json.loads(line)
                max_len = max(len(line['gpt']), max_len)

        with open(osp.join("cv/text_p", f.strip()+'.jsonl'), "r") as fr:
            length = []
            ori_encoding = []
            spt_encoding = []
            for line in fr:
                line = json.loads(line)
                inputs = [line['origin']] + line['gpt']
                output = pipeline(inputs)
                origin_output = np.array(output[0][0])
                origin_output = np.mean(origin_output, axis=0)
                gpt_output = np.zeros((max_len, len(origin_output)))
                for i, out in enumerate(output[1:]):
                    out = np.array(out[0])
                    gpt_output[i] = np.mean(out, axis=0)
                ori_encoding.append(origin_output.tolist())
                spt_encoding.append(gpt_output.tolist())
                length.append(len(line['gpt']))
            np.savez(osp.join("cv/npz", f.strip()), 
                    #  origin_text=line['origin'],
                    #  gpt_text=line['gpt'],
                        len=length,
                    #  origin_encoding_token=output[0],
                        ori_encoding=ori_encoding,
                    #  gpt_encoding_token=output[1:],
                        spt_encoding=spt_encoding,
                    )

                


if __name__ == "__main__":
    from openai import OpenAI
    import os
    os.environ['OPENAI_API_KEY']=''

    client = OpenAI()

    llama_encoding()