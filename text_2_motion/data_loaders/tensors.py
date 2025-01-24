import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    motionbatchTensor = collate_tensors([b['motion'] for b in notnone_batches])
    jointsbatchTensor = collate_tensors([b['joints'] for b in notnone_batches])
    kpsbatchTensor = [b['kps'] for b in notnone_batches]
    
    lenbatchTensor = torch.as_tensor(lenbatch)
    
    maskbatchTensor = lengths_to_mask(lenbatchTensor, motionbatchTensor.shape[1])

    motion = (motionbatchTensor, jointsbatchTensor, kpsbatchTensor)
    
    ori_encodingbatchTensor = torch.stack([b['ori_encoding'] for b in notnone_batches], dim=0)
    spt_encodingbatchTensor = collate_tensors([b['spt_encoding'] for b in notnone_batches])
    spt_lenbatchTensor = torch.as_tensor([b['spt_len'] for b in notnone_batches])
    masksptTensor = lengths_to_mask(spt_lenbatchTensor, spt_encodingbatchTensor.shape[1])
    captionbatch = [b['file_name'] for b in notnone_batches]
    textbatch = [b['text'] for b in notnone_batches]
    tokensbatch = [b['tokens'] for b in notnone_batches]
    cond = {
        'y': {
            'mask': maskbatchTensor, 
            'lengths': lenbatchTensor,
            'ori_encoding': ori_encodingbatchTensor,
            'spt_len': spt_lenbatchTensor,
            'spt_encoding': spt_encodingbatchTensor,
            'spt_mask': masksptTensor,
            'file_name': captionbatch,
            'text': textbatch,
            'tokens': tokensbatch
            }
        }
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'motion': torch.tensor(b[1]).float(),
        'joints': torch.tensor(b[2]).float(),
        'kps': torch.tensor(b[0]).float(),
        'ori_encoding': torch.tensor(b[4]).float(),
        'spt_len': b[5],
        'spt_encoding': torch.tensor(b[6]).float(),
        'lengths': b[3],
        'file_name': b[7],
        'text': b[8],
        'tokens': b[9]
    } for b in batch]
    return collate(adapted_batch)


