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

    # control_info = np.load("/home/wangd/VoxPoser-main/control_info.npy", allow_pickle=True).item()
    control_info = np.load("/home/wangd/GCML/output/custom/control_info0.npy", allow_pickle=True).item()
    n_frames=control_info["n_frames"]
    texts=control_info["texts"]
    hints=control_info["hints"]
    base_position=control_info["base_position"]
    
    joint_ids=control_info["joint_ids"]

    n_frame=n_frames[index]
    joint_id=joint_ids[index]
    text = texts[index:index+1]
    hint = hints[index]        #[1,196,22,3]

    # modify something here when needed to test
    # text=['A person picks up the teacup on the left and then drinks tea.']
    # hint[:,60:110,0,2]=0.5
    # hint[:,60:110,20:22,2]=0.9
    # n_frame=120

    # rearrange xyz axises and set x and y as its negetive because axises are diffent between voxposer and omnicontrol.
    hint=hint[:,:,:,[0,2,1]]        #now is x,z,y in blender when using 
    hint[..., 0] = -hint[..., 0]
    base_position = np.concatenate([base_position,base_position,base_position], axis=0)

    base_position=base_position[:,[0,2,1]]        #now is x,z,y in blender when using 
    base_position[..., 0] = -base_position[..., 0]


 
    return n_frame,text, hint, base_position



last_motion=None
i=0
for i in range(0,1):
    # n_frames,texts,hints,base_position=sequential_generate(index=i, repeat=False)
    # texts=texts+texts+texts
    # hints = np.concatenate([hints,hints,hints], axis=0)
    if i==0:
        texts=["a person sits on the chair","a person sits down","sit on the sofa",]
        # texts=texts+texts+texts
        hints=None
        n_frames=60
    if i==1:
        texts=["a person eats the food in his right hand",]
        texts=texts+texts+texts
        hints=None
        n_frames=60
    if i==2:
        texts=["a person walks forward",]
        texts=texts+texts+texts
        hints=None
        n_frames=60
    out_path=f"save/sample{i}"
    last_motion=generate_motion(n_frames,out_path,texts,hints,i,0)
    render=True
    if render:
        thread0=threading.Thread(target=render_mesh, args=(f"{out_path}/sample00_rep00.mp4",3))
        thread0.start()
        thread1=threading.Thread(target=render_mesh, args=(f"{out_path}/sample01_rep00.mp4",3))
        thread1.start()
        thread2=threading.Thread(target=render_mesh, args=(f"{out_path}/sample02_rep00.mp4",3))
        thread2.start()
thread0.join()
thread1.join()
thread2.join()
for k in range(3):
    folder_path=[]
    for j in range(i+1):
        out_path=f"save/sample{j}/sample0{k}_rep00_obj"
        folder_path.append(f"{out_path}")
    destination_dir = f"save/sample_collection{k}"
    gather_files(folder_path, destination_dir)