from generate import generate_motion
import os
import numpy as np
import torch
import math
from os.path import join as pjoin
from visualize.render_mesh import render_mesh
import threading
from utils.custom import gather_files


def sequential_generate(index=0,repeat=False):

    control_info = np.load("/home/wangd/VoxPoser-main/control_info.npy", allow_pickle=True).item()
    n_frames=control_info["n_frames"]
    texts=control_info["texts"]
    hints=control_info["hints"]

    
    joint_ids=control_info["joint_ids"]

    n_frame=n_frames[index]
    joint_id=joint_ids[index]
    text = texts[index:index+1]
    hint = hints[index]        #[1,196,22,3]


    # rearrange xyz axises and set x and y as its negetive because axises are diffent between voxposer and omnicontrol.
    hint=hint[:,:,:,[0,2,1]]        #now is x,z,y in blender when using 
    hint[..., 0] = -hint[..., 0]



 
    return n_frame,text, hint



last_motion=None
i=0
render=True




round=2

for i in range(0,round):
    n_frames=90
    condition_path=f"sample.npy"
    conditions=np.load(condition_path, allow_pickle=True)
    texts=[]
    for j in range(40):
        item=conditions[j+40*i]
        texts=texts+[item["c_text"]]
    hints = None

    out_path=f"save/sample{i}"
    last_motion=generate_motion(n_frames,out_path,texts,hints,index=-i)

total_results=np.empty((80,196,66))
for i in range(round):
    results=np.load(f"save/sample{i}/results.npy", allow_pickle=True)
    results=results[None][0]
    j=0
    for j in range(results["motion"].shape[0]):
        joints=results["motion"][j].transpose(2,0,1)
        joints=joints[...,[0,2,1]]
        joints[...,0]=-joints[...,0]
        length=joints.shape[0]
        joints=joints.reshape(length,66)
        sample = np.pad(joints, ((0, 196-length), (0, 0)), mode='constant', constant_values=0)
        total_results[j+40*i]=sample
        j+=1
np.save("results.npy", total_results)

total_results=np.empty((80,196,66))
for i in range(round):
    results=np.load(f"save/sample{i}/rot_sample.npy", allow_pickle=True)
    results=results[None][0]
    j=0
    for j in range(results.shape[0]):
        joints=results[j].transpose(2,0,1)
        length=joints.shape[0]
        joints=joints.reshape(length,66)
        sample = np.pad(joints, ((0, 196-length), (0, 0)), mode='constant', constant_values=0)
        total_results[j+40*i]=sample
        j+=1
np.save("results.npy", total_results)