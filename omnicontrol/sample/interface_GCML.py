from generate import generate_motion
import os
import numpy as np
import torch
import math
from os.path import join as pjoin
from visualize.render_mesh import render_mesh
import threading
from utils.custom import gather_files
import shutil

def sequential_generate(index=0,epoch=0,repeat=False):

    control_info = np.load(f"/home/wangd/GCML/output/gcml/control_info{index}.npy", allow_pickle=True).item()
    n_frames=control_info["n_frames"]
    texts=control_info["texts"]
    hints=control_info["hints"]


    n_frame=n_frames

    if len(texts)<=epoch:
        # hints=np.zeros((1,1,90,22,3))
        return n_frame,None,None
    text = texts[epoch]
    hint = hints[epoch]       #[1,196,22,3]  
    if len(hint)==0:
        return n_frame,text,None
    # rearrange xyz axises and set x and y as its negetive because axises are diffent between voxposer and omnicontrol.
    hint=hint[:,:,:,[0,2,1]]        #now is x,z,y in blender when using 
    hint[..., 0] = -hint[..., 0]
 
    return n_frame,text, hint



last_motion=None
i=0
render=True
length=80
results=np.empty((length,60,66))
for j in range(0,3):
    # generate motions without spatial control
    texts=[]
    hints=[]
    text_indices=[]
    hint_indices=[]
    for i in range(0,length):
        n_frames,text,hint=sequential_generate(index=i,epoch=j, repeat=False)
        if (text is not None) and (hint is None):
            texts=texts+[text]
            hints=None
            text_indices.append(i)
    out_path=f"save/sample_text{j}"
    if len(texts)!=0:
        last_motion=generate_motion(60,out_path,texts,hints,index=text_indices,epoch=j)
    # generate motions with spatial control
    texts=[]
    hints=[]
    for i in range(0,length):
        n_frames,text,hint=sequential_generate(index=i,epoch=j, repeat=False)
        if hint is not None:
            texts=texts+[text]
            if len(hints)==0:
                hints=hint
            else:
                if hint.shape[1]>60:
                    hint=hint[:,:60,:,:]
                hints = np.concatenate([hints,hint], axis=0)
            hint_indices.append(i)
    out_path=f"save/sample_hint{j}"
    if len(texts)!=0:
        last_motion=generate_motion(60,out_path,texts,None,index=hint_indices,epoch=j)
    # gather results in sample_hint and sample_text into sample(j)
    if j==0:
        raw_sample=np.zeros((length,263,1,0))
        xyz_sample=np.zeros((length,22,3,0))
    else:
        raw_sample=np.load(f"save/sample{j-1}/raw_sample.npy")
        xyz_sample=np.load(f"save/sample{j-1}/xyz_sample.npy")
    result_raw=np.zeros((length,263,1,raw_sample.shape[-1]+60))
    result_xyz=np.zeros((length,22,3,xyz_sample.shape[-1]+60))
    result_raw[...,:raw_sample.shape[-1]]=raw_sample
    result_xyz[...,:xyz_sample.shape[-1]]=xyz_sample
    if  os.path.exists(f"save/sample_text{j}"):
        raw_sample_text = np.load(f"save/sample_text{j}/raw_sample.npy", allow_pickle=True)
        xyz_sample_text = np.load(f"save/sample_text{j}/xyz_sample.npy", allow_pickle=True)

        for i in range(raw_sample_text.shape[0]):
            result_raw[text_indices[i]]=np.concatenate((raw_sample[text_indices[i]],raw_sample_text[i]),axis=-1)
            result_xyz[text_indices[i]]=np.concatenate((xyz_sample[text_indices[i]],xyz_sample_text[i]),axis=-1)
    if  os.path.exists(f"save/sample_hint{j}"):
        raw_sample_hint = np.load(f"save/sample_hint{j}/raw_sample.npy", allow_pickle=True)
        xyz_sample_hint = np.load(f"save/sample_hint{j}/xyz_sample.npy", allow_pickle=True)
        for i in range(raw_sample_hint.shape[0]):
            result_raw[hint_indices[i]]=np.concatenate((raw_sample[hint_indices[i]],raw_sample_hint[i]),axis=-1)
            result_xyz[hint_indices[i]]=np.concatenate((xyz_sample[hint_indices[i]],xyz_sample_hint[i]),axis=-1)
    if os.path.exists(f"save/sample{j}"):
        shutil.rmtree(f"save/sample{j}")
    os.makedirs(f"save/sample{j}")
    
    np.save(f"save/sample{j}/raw_sample.npy",result_raw)
    np.save(f"save/sample{j}/xyz_sample.npy",result_xyz)
    pass




total_results=np.empty((length,196,66))
results=np.load(f"save/sample{j}/xyz_sample.npy", allow_pickle=True)
for j in range(results.shape[0]):
    joints=results[j].transpose(2,0,1)
    joints=joints[...,[0,2,1]]
    joints[...,0]=-joints[...,0]
    length=joints.shape[0]
    joints=joints.reshape(length,66)
    sample = np.pad(joints, ((0, 196-length), (0, 0)), mode='constant', constant_values=0)
    total_results[j]=sample
mask = (total_results.sum(axis=2) == 0)
np.save("results.npy", total_results)
