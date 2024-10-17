import os
import numpy as np
import torch
import math

from os.path import join as pjoin

# joints:
# 0: pelvis:
# 10: l_foot: kick something
# 11: r_foot: kick something
# 15: head: walk into a tunnel or something
# 20: l_wrist: raise a toolbox, or touch something, or hold something
# 21: r_wrist: raise a toolbox, or touch something, or hold something, or walk with one hand on the handrail

# -sigmoid maps a real number between (0,1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unnatural_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person walks',
        'a person walks',
    ]
    control = [
        # pelvis
        spiral_forward(n_frames),
        specify_points(n_frames, [[0, 0.0, 0.9, 0.0], [1, 0.0, 0.9, 2.5]]),
    ]

    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)

    # extend for example 1
    control[1, 1:195] = control[1, 1]

    control = control[index:index+1]
    text = text[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id



def motion_inbetweening(n_frames=120, raw_mean=None, raw_std=None, index=0):
    '''
    Trajectories are load from the HumanML3D dataset
    '''
    from data_loaders.humanml.data.dataset import HumanML3D
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    dataset = HumanML3D(mode='eval')
    control_full = []
    texts_full = []

    _, _, text, _, motion, _, _, _  = dataset[index]
    motion = dataset.t2m_dataset.inv_transform(motion)
    n_joints = 22 if motion.shape[-1] == 263 else 21
    joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
    joints = joints.numpy()
        
    # normalize
    joints = (joints - raw_mean.reshape(1, 22, 3)) / raw_std.reshape(1, 22, 3)
        
    mask = np.zeros((n_frames, 22, 1))
    control_joints = [0, 10, 11, 15, 20, 21]
    for j in control_joints:
        mask[[0, -1], j] = 1
        
    control = joints * mask
    control = control.reshape((n_frames, -1))
    control_full.append(control)
    texts_full.append(text)

    control_full = np.stack(control_full)
    return texts_full, control_full, None


def combination_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    '''
    Here I just provide one simple example for combination contorlling (l_wrist, r_wrist)
    Need to optimize if want to add more...
    '''
    text = [
        # l_wrist
        'a person raises the toolbox with both hands',
        'a person walks forward',
        'a person puts their hands on something and attempts to push it',
    ]
    control = [
        # l_wrist, r_wrist
        [straight_fb(n_frames, indices=[1, 2], scale=0.5, x_offset=0.15, y_offset=0.6, z_offset=0.5),
        straight_fb(n_frames, indices=[1, 2], scale=0.5, x_offset=-0.15, y_offset=0.6, z_offset=0.5)],
        [straight(n_frames, indices=[2, 0], scale=4.0, x_offset=0, y_offset=1.2, z_offset=0.0),
        straight(n_frames, indices=[2, 0], scale=4.0, x_offset=-0.3, y_offset=1.0, z_offset=0.0)],
        [specify_points(n_frames, [[50, 0.18, 1.0, 0.8]]),
        specify_points(n_frames, [[50, -0.18, 1.0, 0.8]])]
    ]
    joint_id = np.array([
        [20, 21],
        [15, 21],
        [20, 21],
        ])
    control = np.stack(control)

    # extend for example 2
    control[2, :, 50:90] = control[2, :, 50:51]

    text = text[index:index+1]
    control = control[index:index+1]
    joint_id = joint_id[index:index+1]

    # normalize
    # control = (control - raw_mean.reshape(22, 1, 3)) / raw_std.reshape(22, 1, 3)
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((control.shape[0], n_frames, -1))
    return text, control_full, joint_id


def pelvis_dense_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person plays a violin with their left hand in the air and their right hand holding the bow',
        'a person plays a violin with their left hand in the air and their right hand holding the bow',
        'a person runs',
        'a person holds a box and walks',
        'a person walks like a duck',
        'a person carries a box at the front of him',
        'A person is skipping rope.'
    ]
    control = [
        # pelvis
        # circle
        circle(n_frames, r=0.8, indices=[0, 2], x_offset=0.0, y_offset=0.8, z_offset=0.0),
        # short s
        s_line(n_frames, indices=[2, 0], x_offset=0.0, y_offset=0.8, z_offset=0.0),
        # straight line
        straight_diagonal_uniform(n_frames, indices=[0, 2], scale=9.0, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # long s
        s_line_long(n_frames, indices=[2, 0], scale=1/3, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # duck walk
        straight_diagonal_uniform(n_frames, indices=[0, 2], scale=4, x_offset=0.0, y_offset=0.6, z_offset=0.0),
        # step walk
        straight_forward_step_uniform(n_frames, indices=[0, 2], scale=4, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # step walk
        circle(n_frames, r=0.8, indices=[0, 2], x_offset=0.0, y_offset=0.9, z_offset=0.0),
    ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)

    # sparsify
    # early stop for case 2
    control[2, 140:] = control[2, 139:140]

    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        # -sum the numbers at the last axis, compare them with 0 to get a bool array, so mask
        # -are frames where no control signal exists
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id


def pelvis_sparse_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'A person walks towards a chair, sits down, and then walks back.',
        'a person holds a box over his head and walks',
        'a person is leaving at someone with his left hand',
        'A person runs to the right then runs to the left.',
    ]
    control = [
        # pelvis
        specify_points(n_frames, [[100, 1.0, 0.58, 1.2]]),
        specify_points(n_frames, [[30, 0.55, 0.9, 1.6], [60, -0.55, 0.9, 3.2],[90, 0.55, 0.9, 4.8],[120, -0.55, 0.9, 6.4],[150, 0.55, 0.9, 8.0],[180, -0.55, 0.9, 9.6]]),
        s_line_long(n_frames, indices=[2, 0], scale=1/3, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # 30-78, 117 - 156
        specify_points(n_frames, [[30, -2, 0.6, 1.0], [117, 2, 0.6, 1.0]]),
    ]

    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)

    # extend for case 0
    control[0, 90:110] = control[0, 100]
    # sparsify for example 2
    control[2, 0:20] = 0
    control[2, 60:180] = 0
    # extend for example 3
    control[3, 30:78] = control[3, 30]
    control[3, 117:156] = control[3, 117]

    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id



def wrist_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # wrist
        'a person puts hands on the armrest',
        'a person picks the cups and puts it on another table',
        'a person walks forward, bends down to pick something up off the ground.'
        'A person walks forward, then picks up a cup.',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
        'a person raises the toolbox with the use of one hand',
    ]
    control = [
        # pelvis
        straight_forward_uniform(n_frames, indices=[2], scale=8, x_offset=-0.4, y_offset=1.35, z_offset=0.0),
        specify_points(n_frames, [[80, -0.15, 0.6, 1.2], [160, 1.8, 0.6, 1.2]]),
        specify_points(n_frames, [[160, 0, 0.2, 2.0]]),
        circle(n_frames, r=0.01, indices=[0, 2], x_offset=-0.15, y_offset=0.8, z_offset=1.5),
        # ---
        straight_fb(n_frames, indices=[1, 2], scale=0.5, x_offset=0.5, y_offset=0.6, z_offset=0.5),
        straight_fb(n_frames, indices=[1, 2], scale=0.5, x_offset=0, y_offset=0.6, z_offset=0.5),
        straight_fb(n_frames, indices=[1, 2], scale=0.5, x_offset=0.5, y_offset=0.6, z_offset=0.0),
        straight(n_frames, indices=[1, 2], scale=-1, x_offset=0.0, y_offset=1.15, z_offset=0.5),
        straight(n_frames, indices=[1, 2], scale=-1, x_offset=0.5, y_offset=1.15, z_offset=0.0),
        straight(n_frames, indices=[1, 2], scale=-1, x_offset=0.5, y_offset=1.15, z_offset=0.5),
        circle(n_frames, r=0.01, indices=[0, 2], x_offset=0.0, y_offset=0.5, z_offset=0.0),
        circle(n_frames, r=0.01, indices=[0, 2], x_offset=0.5, y_offset=0.5, z_offset=0.5),
        circle(n_frames, r=0.01, indices=[0, 2], x_offset=0.5, y_offset=1.6, z_offset=0.5),
    ]
    joint_id = np.array([
        # wrist 
        21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        ])
    control = np.stack(control)

    # extend for example 1
    # control[-1, 90:110] = control[-1, 100]
    # sparsify for example 3
    # control[-1, 0:20] = 0
    # control[-1, 60:180] = 0
    # extend for example 1
    control[1, 50:90] = control[1, 80]
    control[1, 130:196] = control[1, 160]
    # extend for example 2
    control[2, 150:180] = control[2, 160]
    # sparsify for example 3
    control[3, :78] = 0
    control[3, 119:] = 0


    text = text[index:index+1]
    control = control[index:index+1]
    joint_id = joint_id[index:index+1]


    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id


def head_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # head
        'slowly and cautiously walking forward while holding something to the sides of the person.',
        'a person walks.',
        'a person keeps jumping until his head reaches the target',
        'a person is walking and has injured their leg',
        'a person bows',
    ]
    control = [
        # head
        straight_forward_uniform(n_frames, indices=[2], scale=5, x_offset=0, y_offset=1.5, z_offset=0.0),
        # long s
        s_line_long(n_frames, indices=[2, 0], scale=1/3, x_offset=0.0, y_offset=1.65, z_offset=0.0),
        # jump
        specify_points(n_frames, [[150, 0, 1.9, 2.0]]),
        circle(n_frames, r=0.7, indices=[0, 2], x_offset=0.0, y_offset=1.6, z_offset=0.0),
        specify_points(n_frames, [[90, 0, 0.8, 1.0]]),
    ]
    joint_id = np.array([
        # head 
        15,
        ])
    control = np.stack(control)

    # extend for example 4
    control[4, 90:110] = control[4, 100]

    control = control[index:index+1]
    text = text[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id


def foot_text_control_example(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # foot
        'A person stands and kicks with his left leg.',
        'A person hops three times',
        'a person is prancing with their arm raised as if playing an instrument.'
        'A person stands and kicks with his right leg.',
    ]
    control = [
        # left foot
        specify_points(n_frames, [[30, 0, 1.0, 1.0]]),
        # right foot
        specify_points(n_frames, [[40, 0, 0, 1], [100, 0.0, 0, 2], [155, 0.0, 0, 3]]),
        # right foot
        specify_points(n_frames, [[40, 1, 0, 1], [100, 1, 0, 2], [155, 2, 0, 2]]),
        # right foot
        specify_points(n_frames, [[30, 0, 1.0, 1.0]]),
    ]
    joint_id = np.array([
        # right foot 
        10, 11, 11, 11
        ])
    control = np.stack(control)

    # extend for example 0, 1 and 2, 3
    control[0, 20:40] = control[0, 30]
    control[1, 35:55] = control[1, 40]
    control[1, 80:100] = control[1, 100]
    control[1, 135:155] = control[1, 155]
    control[2, 35:55] = control[2, 40]
    control[2, 80:100] = control[2, 100]
    control[2, 135:155] = control[2, 155]
    control[3, 20:40] = control[0, 30]

    control = control[index:index+1]
    text = text[index:index+1]
    joint_id = joint_id[index:index+1]
    
    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id

# -draw a circle, indices means axis, returns a list of coordinates[n_frames,3]
def circle(n_frames=120, r=0.8, indices=[0, 2], x_offset=0.5, y_offset=0.9, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_circle(n_frames, r)
    hint [:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight(n_frames=120, indices=[1, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward(n_frames, scale=scale)
    hint [:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def specify_points(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    for point in points:
        hint[point[0]] = point[1:]
    # points = sample_points_forward_uniform(n_frames, scale=scale)
    # hint[:, indices] = np.array(points)[..., np.newaxis]
    # hint[:, 0] += x_offset
    # hint[:, 1] += y_offset
    # hint[:, 2] += z_offset
    return hint

def specify_points_all(n_frames=120, points=[[1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    for point in points:
        hint[:,0] = point[1:]
    # points = sample_points_forward_uniform(n_frames, scale=scale)
    # hint[:, indices] = np.array(points)[..., np.newaxis]
    # hint[:, 0] += x_offset
    # hint[:, 1] += y_offset
    # hint[:, 2] += z_offset
    return hint


def spiral_forward(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    # how many times:
    n = 3
    # radius
    r = 0.3
    # offset
    o = 0.9
    # angle step size
    angle_step = 2*np.pi/ (n_frames / n)

    points = []

    start_from = - np.pi / 2

    for i in range(n_frames):
        theta = i * angle_step + start_from

        x = r*np.cos(theta)
        y = r*np.sin(theta) + o
        z = i * 0.02

        points.append((x, y, z))

    hint = np.stack(points)
    return hint


def straight_diagonal_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_step_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5, step_ratio=0.5):
    hint = np.zeros((n_frames, 3))
    sub_frame = int(n_frames * step_ratio)
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[sub_frame:, 1] -= 0.2
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_backward_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 2
    points = sample_points_forward_uniform(sub_frame, scale=scale)
    hint[:sub_frame, indices] = np.array(points)[..., np.newaxis]
    hint[sub_frame:, indices] = np.array(points[::-1])[..., np.newaxis]
    hint[sub_frame:, 0] += 0.5
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_fb(n_frames=120, indices=[1, 2], scale=0.5, x_offset=0.5, y_offset=0.6, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_back_verticel(n_frames)
    hint [:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] *= scale
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def s_line(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_s(n_frames)
    hint [:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def s_line_long(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5, scale=1/3, scale1=2/3):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 3
    for i in range(3):
        points = sample_points_s(sub_frame, scale, scale1)
        hint[sub_frame * i:sub_frame * (i + 1), indices] = points
        hint[sub_frame * i:sub_frame * (i + 1), 0] += x_offset
        hint[sub_frame * i:sub_frame * (i + 1), 1] += y_offset
        hint[sub_frame * i:sub_frame * (i + 1), 2] += z_offset + 2 * scale * i * np.pi
    return hint


def s_line_middlelong(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5, scale=1/3, scale1=1):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 2
    for i in range(2):
        points = sample_points_s(sub_frame, scale, scale1)
        hint[sub_frame * i:sub_frame * (i + 1), indices] = points
        hint[sub_frame * i:sub_frame * (i + 1), 0] += x_offset
        hint[sub_frame * i:sub_frame * (i + 1), 1] += y_offset
        hint[sub_frame * i:sub_frame * (i + 1), 2] += z_offset + 2 * scale * i * np.pi
    return hint


def sample_points_circle(n, r=0.8):
    # number of points

    # angle step size
    angle_step = 2*np.pi/n

    points = []

    start_from = - np.pi / 2

    for i in range(n):
        theta = i * angle_step + start_from

        x = r*np.cos(theta)
        y = r*np.sin(theta) + r

        points.append((x, y))
    return points


def sample_points_s(n, scale=1/3, scale1=1):
    # number of points

    # angle step size
    angle_step = 2*np.pi/n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = theta * scale
        y = np.sin(theta) * scale1

        points.append((x, y))
    return points


def sample_points_forward(n, scale=1.0):
    # number of points

    # angle step size
    angle_step = np.pi/n/2

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = np.sin(theta) * scale
        y = 0

        points.append((x, y))
    return points


def sample_points_forward_uniform(n, scale=1.0):
    # angle step size
    step = scale / n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * step + start_from

        x = theta

        points.append(x)
    return points



def sample_points_forward_back_verticel(n):
    # number of points

    # angle step size
    angle_step = 2 * np.pi/n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = np.cos(theta)
        y = 0

        points.append((x, y))
    return points


def pelvis_control(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person plays badminton',
        'a frightened person runs away',
        'a person runs',
        'a person holds a box and walks',
        'a person walks like a duck',
        'a person carries a box at the front of him',
        'A person is skipping rope.'
    ]
    control = [
        # pelvis
        # circle
        circle(n_frames, r=0.8, indices=[0, 2], x_offset=0.0, y_offset=0.8, z_offset=0.0),
        # short s
        s_line(n_frames, indices=[2, 0], x_offset=0.0, y_offset=0.8, z_offset=0.0),
        # straight line
        straight_diagonal_uniform(n_frames, indices=[0, 2], scale=9.0, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # long s
        s_line_long(n_frames, indices=[2, 0], scale=1/3, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # duck walk
        straight_diagonal_uniform(n_frames, indices=[0, 2], scale=4, x_offset=0.0, y_offset=0.6, z_offset=0.0),
        # step walk
        straight_forward_step_uniform(n_frames, indices=[0, 2], scale=4, x_offset=0.0, y_offset=0.9, z_offset=0.0),
        # step walk
        circle(n_frames, r=0.8, indices=[0, 2], x_offset=0.0, y_offset=0.9, z_offset=0.0),
    ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)

    # sparsify
    # early stop for case 2
    control[2, 140:] = control[2, 139:140]

    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        # -sum the numbers at the last axis, compare them with 0 to get a bool array, so mask
        # -are frames where no control signal exists
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id



def joints_avoidance(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person walks forward',
        'a frightened person runs away',
        'a person runs'
    ]


    control = [
        # sprcify
        [specify_points(n_frames, [[90, 0, 0.8, 0.5]])],
        # circle
        [circle(n_frames, r=0.2, indices=[0, 2], x_offset=0.0, y_offset=2.0, z_offset=0.0)],
        # short s
        [s_line(n_frames, indices=[2, 0], x_offset=0.0, y_offset=2.0, z_offset=0.0)],    ]
    joint_id = np.array([
        # pelvis 
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        ])
    for i in range(21):
        control[0].append(control[0][0])
        control[1].append(control[1][0])
        control[2].append(control[2][0])
    control = np.stack(control)

    # sparsify
    # early stop for case 2
    control[2, 140:] = control[2, 139:140]

    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_
    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id

def pelvis_avoidance(n_frames=120, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person walks forward',
        'a person walks forward',
        'a person runs forward'
    ]


    control = [
        # sprcify
        [specify_points(n_frames, [[90, 0, 0.8, 0.5]])],
        # circle
        [s_line(n_frames, indices=[2, 0], x_offset=0.0, y_offset=2.0, z_offset=0.0)],
        # short s
        [s_line(n_frames, indices=[2, 0], x_offset=0.0, y_offset=2.0, z_offset=0.0)]
        ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)


    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_
    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id

def specify_every(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    for point in points:
        hint[point[0]] = point[1:]
    return hint

def self_defined_control(n_frames=196, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person stand up from a chair and then walks forward to open the door',
        'A person keeps walking forward for a long time.',
        'A person turns around on the spot and then walks forward.'

    ]
    forward = np.zeros((n_frames, 3))
    right = np.zeros((n_frames, 3))
    backward = np.zeros((n_frames, 3))
    forward[10]=[0,0.4,1]
    forward[20]=[1,0.8,4]
    forward[40]=[-1,0.8,8]
    forward[80]=[2,0.8,12]
    forward[120]=[2,0.8,20]
    right[10]=[1,0.8,0]
    right[20]=[4,0.8,2]
    right[40]=[8,0.4,-2]
    right[80]=[8,0.4,2]
    right[120]=[16,0.4,2]
    backward[10]=[0,0.8,0]
    backward[40]=[0,0.8,-3]
    backward[80]=[0,0.8,-5]
    control = [
        [forward],
        [right],
        [backward]
        ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)  

    text = text[index:index+1]
    control = control[index:index+1]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control_ * mask[..., np.newaxis]
        control_full[i, :, joint_id[i], :] = control_

    control_full = control_full.reshape((len(control), n_frames, -1))
    return text, control_full, joint_id


def self_defined_inbetween(n_frames=120, raw_mean=None, raw_std=None, index=0):
    '''
    Trajectories are load from the HumanML3D dataset
    '''
    text = [
        # pelvis
        'a person stand up from a chair and then walks forward to open the door',
        'A person keeps walking forward for a long time.',
        'A person turns around on the spot and then walks forward.'

    ]

    control_full = []
    texts_full = []
    text = text[index:index+1]
        
    file_path='save/omnicontrol_ckpt/samples_omnicontrol_ckpt__humanml3d_seed10_predefined/results.npy'
    last_result = np.load(file_path, allow_pickle=True).item()
    all_motions=last_result['motion']       #[rep*sap,22,3,196]
    motion=all_motions[index*2]
    joints=np.transpose(motion,(2,0,1))
    mask = np.zeros((n_frames, 22, 1))      
    control_joints = [0, 10, 11, 15, 20, 21]
    for j in control_joints:
        mask[[0, -1], j] = 1
        
    control = joints * mask     #[196*66]
    control = control.reshape((n_frames, -1))
    control_full.append(control)

    control_full = np.stack(control_full)
    return text, control_full, None

def sequential_generate(n_frames=196, raw_mean=None, raw_std=None, index=0):
    text = [
        # pelvis
        'a person walks forward',
        'A person turns around on the spot and sit down.',
        'A person stand up from a chair.'

    ]

    forward = np.zeros((n_frames, 3))
    sit = np.zeros((n_frames, 3))
    stand = np.zeros((n_frames, 3))
    forward[20]=[0,0.4,1]
    forward[40]=[0.5,0.8,4]
    forward[80]=[-0.5,0.8,8]
    forward[120]=[1,0.8,12]
    sit[20]=[0,0.8,0]
    sit[40]=[0,0.8,0]
    sit[80]=[0,0.4,0]
    sit[120]=[0,0.4,0]
    stand[40]=[0,0.4,0]
    stand[80]=[0,0.8,0]
    stand[120]=[0,0.8,0]
    control = [
        [forward],
        [sit],
        [stand]
        ]
    joint_id = np.array([
        # pelvis 
        0,
        ])
    control = np.stack(control)  


    text = text[index:index+1]
    control = control[index:index+1]        #[1,1,196,3]

    # normalize
    control_full = np.zeros((len(control), n_frames, 22, 3)).astype(np.float32)
    for i in range(len(control)):
        mask = control[i].sum(-1) != 0
        control_ = (control[i] - raw_mean.reshape(22, 1, 3)[joint_id[i]]) / raw_std.reshape(22, 1, 3)[joint_id[i]]
        control_ = control * mask[..., np.newaxis]     #[1,196,3]
        control_full[i, :, joint_id[i], :] = control_   #[1,196,22,3]

    control_full = control_full.reshape((len(control), n_frames, -1))   #[1,196,66]
    # modify the first frame to be the same as last generated motion to ensure consistence
    file_path='save/omnicontrol_ckpt/samples_omnicontrol_ckpt__humanml3d_seed10_predefined/results.npy'
    if os.path.exists(file_path):
        last_result = np.load(file_path, allow_pickle=True).item()
        all_motions=last_result['motion']       #[rep*sap,22,3,196]
        motion=all_motions[0]
        joints=np.transpose(motion,(2,0,1))
        # substract all joints with the pelvis joint
        pelvis_position=joints[:,0,:]
        pelvis_position=pelvis_position[:,np.newaxis,:]
        joints=joints-pelvis_position
        mask = np.zeros((joints.shape[0], 22, 1))      
        control_joints = [0, 10, 11, 15, 20, 21]
        for j in control_joints:
            mask[[0, -1], j] = 1
            
        inbetween_control = joints * mask     #[196*22*3]
        inbetween_control = inbetween_control.reshape((joints.shape[0], -1))  #[196*66]
        control_full[0,0,:]=inbetween_control[-1,:]

    return text, control_full, joint_id

def collate_all(n_frames, dataset):
    if dataset == 'humanml':
        spatial_norm_path = './dataset/humanml_spatial_norm'
    elif dataset == 'kit':
        spatial_norm_path = './dataset/kit_spatial_norm'
    else:
        raise NotImplementedError('unknown dataset')
    raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
    raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

    texts0, hints0, _ = self_defined_control(n_frames, raw_mean, raw_std, index=0)
    texts1, hints1, _ = self_defined_control(n_frames, raw_mean, raw_std, index=1)
    # texts2, hints2, _ = self_defined_control(n_frames, raw_mean, raw_std, index=2)
    texts2, hints2, _ = self_defined_control(n_frames, raw_mean, raw_std, index=2)
    texts = texts0+texts1+texts2
    hints = np.concatenate([hints0,hints1,hints2], axis=0)
    return texts, hints
