import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import smplx
from tqdm import tqdm
    
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    # r_rot_quat = torch.nan_to_num(r_rot_quat, nan=0.0)
    # r_pos = torch.nan_to_num(r_pos, nan=0.0)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    # positions = torch.nan_to_num(positions, nan=0.0)
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    # positions = torch.nan_to_num(positions, nan=0.0)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    # positions = torch.nan_to_num(positions, nan=0.0)
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    
    return positions

class JOI2KP(nn.Module):
    
    def __init__(self, input_type='smplx'):
        super().__init__()
        self.input_type = input_type.lower()
        meta       = joblib.load('Kinematic-Phrases/sample/meta_info.pkl')
        
        if self.input_type in ['smplx']:
            JOINT_NAMES = [
                "pelvis",
                "left_hip",
                "right_hip",
                "spine1",
                "left_knee",
                "right_knee",
                "spine2",
                "left_ankle",
                "right_ankle",
                "spine3",
                "left_foot",
                "right_foot",
                "neck",
                "left_collar",
                "right_collar",
                "head",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "jaw",
                "left_eye",
                "right_eye",
            ]
            joint_idx = {' '.join(k.split('_')): v for v, k in enumerate(JOINT_NAMES)}
            joint_idx['left eye']  = 14
            joint_idx['right eye'] = 15
        else:
            raise NotImplementedError
            
        axis_idx  = {
            'ud': 0,
            'rl': 1, 
            'fb': 2,
            'none': 3, 
            'left upper arm': 4,
            'left thigh': 5,
            'right upper arm': 6, 
            'right thigh': 7,
        }
        limbs = {
            'left lower arm': ('left wrist', 'left elbow'), 
            'left upper arm': ('left shoulder', 'left elbow'), 
            'left shank':     ('left foot', 'left knee'), 
            'left thigh':     ('left hip', 'left knee'), 
            'left body':      ('left shoulder', 'left hip'), 
            'right lower arm': ('right wrist', 'right elbow'), 
            'right upper arm': ('right shoulder', 'right elbow'), 
            'right shank':     ('right foot', 'right knee'), 
            'right thigh':     ('right hip', 'right knee'), 
            'right body':      ('right shoulder', 'right hip'), 
            'upper body':      ('pelvis', 'neck'),
        }
        fullLimbs = {
            'left arm':        ('left lower arm', 'left upper arm'), 
            'left leg':        ('left shank', 'left thigh'), 
            'left upper arm':  ('left body', 'left upper arm'),
            'left thigh':      ('upper body', 'left thigh'),
            'right arm':       ('right lower arm', 'right upper arm'), 
            'right leg':       ('right shank', 'right thigh'),
            'right upper arm': ('right body', 'right upper arm'),
            'right thigh':     ('upper body', 'right thigh'),
        }
        idx = [] # each item, j1 index, j2 index, axis index
        for i, info in enumerate(meta['IDX2META']):
            if info[0] == 'pp':
                # e.g.
                # (left hand, ud)
                # 1: left hand moves upwards
                # -1: left hand moves downwards
                part, ax = info[1] # 1: part moving upwards ()
                idx.append([joint_idx[part], 0, axis_idx[ax]])
            elif info[0] == 'pdp': 
                # e.g.
                # (left hand, right hand)
                # 1: lhand and rhand moves away from each other
                # -1: lhand and rhand moves closer
                ja, jb = info[1]
                idx.append([joint_idx[ja], joint_idx[jb], axis_idx['none']])
            elif info[0] in ['prpp', 'lop']:
                # e.g.
                # (left hand, right hand, ud)
                # 1: lhand above rhand
                # -1: lhand below rhand
                ja, jb, ax = info[1]
                idx.append([joint_idx[ja], joint_idx[jb], axis_idx[ax]])
            elif info[0] == 'lap':
                # e.g.
                # (left arm)
                # 1: left arm unbends
                # -1: left arm bends
                fLimb = fullLimbs[info[1]]
                idx.append([joint_idx[limbs[fLimb[0]][0]], joint_idx[limbs[fLimb[0]][1]], axis_idx[fLimb[1]]])

        self.idx = np.array(idx)
        self.joint_idx = joint_idx
    
    def forward(self, joi, index=None):
        # joints are filled in batches, so we need to calculate batch first
        with torch.no_grad():
            axis = torch.zeros(joi.shape[0], joi.shape[1], 8, 3, device=joi.device)
            axis[:, :, 0, -1] = 1. # ud
            axis[:, :, 1] = joi[:, :, self.joint_idx['right hip']] - joi[:, :, self.joint_idx['left hip']]
            axis[:, :, 2] = torch.cross(axis[:, :, 0], axis[:, :, 1])
            # axis[:, :, 2] = 1.
            axis[:, :, 3] = 1.
            axis[:, :, 4] = joi[:, :, self.joint_idx['left shoulder']]  - joi[:, :, self.joint_idx['left elbow']]
            axis[:, :, 5] = joi[:, :, self.joint_idx['left hip']]    - joi[:, :, self.joint_idx['left knee']]
            axis[:, :, 6] = joi[:, :, self.joint_idx['right shoulder']] - joi[:, :, self.joint_idx['right elbow']]
            axis[:, :, 7] = joi[:, :, self.joint_idx['right hip']]   - joi[:, :, self.joint_idx['right knee']]
            axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-5)
            # axis = axis.clone().detach()
        if index is None:
            ind1 = torch.sum((joi[:, :, self.idx[:381, 0]] - joi[:, :, self.idx[:381, 1]]) * axis[:, :, self.idx[:381, 2]], axis=-1)
            ind2 = torch.arccos(torch.sum((joi[:, :, self.idx[381:, 0]] - joi[:, :, self.idx[381:, 1]]) * axis[:, :, self.idx[381:, 2]] / (torch.norm((joi[:, :, self.idx[381:, 0]] - joi[:, :, self.idx[381:, 1]]), dim=-1, p=2, keepdim=True) + 1e-5), dim=-1))
            ind3 = torch.sum((joi[:, 1:, [0, 0, 0]] - joi[:, :-1, [0, 0, 0]]) * axis[:, :-1, :3], dim=-1)
            ind3 = torch.cat([ind3, ind3[:, -1:]], dim=1)
            indicators = torch.cat([ind1, ind2, ind3], axis=2)
            indicators[:, 1:, :115] = indicators[:, 1:, :115] - indicators[:, :-1, :115]
            indicators[:, :1, :115] = indicators[:, 1:2, :115]
            indicators[:, 1:, 381:389]  = indicators[:, 1:, 381:389] - indicators[:, :-1, 381:389]
            indicators[:, :1, 381:389]  = indicators[:, 1:2, 381:389]
            indicators = torch.tanh(indicators)
        return indicators

        # axis = torch.zeros(joi.shape[0], 8, 3, device=joi.device)
        # axis[:, 0, -1] = 1. # ud
        # axis[:, 1] = joi[:, self.joint_idx['right hip']] - joi[:, self.joint_idx['left hip']] # rl
        # axis[:, 2] = torch.cross(axis[:, 0], axis[:, 1]) # fb
        # axis[:, 3] = 1. # none
        # axis[:, 4] = joi[:, self.joint_idx['left shoulder']]  - joi[:, self.joint_idx['left elbow']]  # left upper arm
        # axis[:, 5] = joi[:, self.joint_idx['left hip']]    - joi[:, self.joint_idx['left knee']]   # left thigh
        # axis[:, 6] = joi[:, self.joint_idx['right shoulder']] - joi[:, self.joint_idx['right elbow']] # right upper arm
        # axis[:, 7] = joi[:, self.joint_idx['right hip']]   - joi[:, self.joint_idx['right knee']]  # right thigh
        # axis = axis / torch.norm(axis, p=2, dim=2, keepdim=True)
        # if index is None:
        #     ind1 = torch.sum((joi[:, self.idx[:381, 0]] - joi[:, self.idx[:381, 1]]) * axis[:, self.idx[:381, 2]], axis=-1)
        #     ind2 = torch.arccos(torch.sum((joi[:, self.idx[381:, 0]] - joi[:, self.idx[381:, 1]]) * axis[:, self.idx[381:, 2]] / (torch.norm((joi[:, self.idx[381:, 0]] - joi[:, self.idx[381:, 1]]), dim=2, p=2, keepdim=True) + 1e-8), dim=-1))
        #     ind3 = torch.sum((joi[1:, [0, 0, 0]] - joi[:-1, [0, 0, 0]]) * axis[:-1, :3], dim=-1)
        #     ind3 = torch.cat([ind3, ind3[-1:]])
        #     indicators = torch.cat([ind1, ind2, ind3], axis=1)
        #     indicators[1:, :115] = indicators[:, :115][1:] - indicators[:, :115][:-1]
        #     indicators[:1, :115] = indicators[1:2, :115]
        #     indicators[1:, 381:389]  = indicators[:, 381:389][1:] - indicators[:, 381:389][:-1]
        #     indicators[:1, 381:389]  = indicators[1:2, 381:389]
        #     indicators[torch.abs(indicators) < 1e-3] = 0
        #     indicators = torch.sign(indicators)
        #     return indicators
        # else:
        #     if index < 381:
        #         indicator = torch.sum((joi[:, self.idx[index, 0]] - joi[:, self.idx[index, 1]]) * axis[:, self.idx[index, 2]], axis=-1)
        #     elif index < 389:
        #         x1 = joi[:, self.idx[index, 0]] - joi[:, self.idx[index, 1]]
        #         x2 = axis[:, self.idx[index, 2]]
        #         cos = torch.clip(torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8), -1, 1)
        #         indicator = torch.arccos(cos)
        #     else:
        #         indicator = torch.sum((joi[1:, 0] - joi[:-1, 0]) * axis[:-1, index - 389], dim=-1)
        #     if index < 115 or 389 > index > 381:
        #         indicator = torch.diff(indicator)
        #     indicator[torch.abs(indicator) < 1e-3] = 0
        #     indicator = torch.sign(indicator)
        #     return indicator

def process_joint_files(joint_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 实例化 JOI2KP 模型
    kp = JOI2KP()

    # 获取 joint_dir 目录中的所有 .npy 文件
    joint_files = [f for f in os.listdir(joint_dir) if f.endswith('.npy')]
    
    cnt = 0
    skipped_files = []  # 用于记录跳过的文件

    for joint_file in tqdm(joint_files):
        cnt += 1
        # 读取 .npy 文件中的数据
        joint_path = os.path.join(joint_dir, joint_file)
        joints = np.load(joint_path)

        joints_tensor = torch.tensor(joints, dtype=torch.float32)
                
        # 如果 joints_tensor 是一个 2D 张量，扩展其第一个维度为 1
        if joints_tensor.dim() == 2:
            skipped_files.append(joint_file)  # 记录跳过的文件
            continue

        joints_tensor = joints_tensor.reshape(1, joints_tensor.shape[0], joints_tensor.shape[1], joints_tensor.shape[2])
        
        # 将数据传递给 kp(joints)
        kp_output = kp(joints_tensor)

        # 将输出转换为 numpy 数组
        kp_output_np = kp_output.detach().cpu().numpy()
               
        # 构造输出文件名
        output_file = f"{joint_file}"
        output_path = os.path.join(output_dir, output_file)

        # 保存输出到 .npy 文件中
        np.save(output_path, kp_output_np)

        # print(f"Processed {joint_file} and saved to {output_file}")
        # exit(0)

    # 将跳过的文件写入 skip.txt
    if skipped_files:
        with open('skip.txt', 'w') as f:
            for file in skipped_files:
                f.write(f"{file}\n")
        print(f"Skipped files written to skip.txt: {len(skipped_files)} files.")

if __name__ == '__main__':
    joint_dir = '../dataset/HumanML3D/new_joints/'
    output_dir = '../dataset/HumanML3D/kp_joint_vecs/'

    process_joint_files(joint_dir, output_dir)
