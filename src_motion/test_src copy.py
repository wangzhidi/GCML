import numpy as np
from env_utils import execute, specify_pelvis_trajectory, specify_joint_position, specify_pelvis_position, generate_motion
from perception_utils import parse_query_obj
from plan_utils import get_affordance_map, get_avoidance_map, get_velocity_map, get_rotation_map, get_gripper_map

# Query: a person walks forward for 5.0 meters.
control_text='a person walks forward'
n_frames=120
target=parse_query_obj('person')
affordance_map=get_affordance_map(f'a point 500cm in front of {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: a person walks forward to sit on a chair.
control_text='a person walks forward and then sit on the chair'
n_frames=120
target=parse_query_obj('chair')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)


# Query: a person walks happily to the bed.
control_text='a person walks happily'
n_frames=120
target=parse_query_obj('bed')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: a person walks to the chair while avoiding collision with the bed.
control_text='a person walks'
n_frames=180
target=parse_query_obj('chair')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'2 meters away from the bed')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: a person walks forward to sit on a chair, watch out for the bed.
control_text='a person walks forward'
n_frames=180
target=parse_query_obj('chair')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'2 meters away from the bed')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: A person walks forward to the chair near the piano.
control_text='a person walks forward'
n_frames=180
target=parse_query_obj('chair near the piano')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)


# Query: a person playing piano.
control_text='a person playing piano'
n_frames=120
control_frames=[]
control_hints=[]
control_frames=range(20,120)
control_hints=parse_query_obj("seat of the chair").position_world
pelvis_height=0.5 # when sitting down, the pelvis height is usually 0.5
#For complex motion generation, do not specify other joints
specify_pelvis_position(control_text, control_frames, control_hints,pelvis_height, n_frames)
generate_motion(control_text, n_frames)

# Query: A tired person walks to the fruit.
control_text='A tired person walks.'
n_frames=180
target=parse_query_obj('fruit')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: A person picks up fruit from the table.
control_text='A person picks up fruit from the table.'
n_frames=90
control_joints=["pelvis","right_hand"]
control_frames={}
control_hints={}
control_frames["pelvis"]=range(0,90)
control_hints["pelvis"]=parse_query_obj("last created motion frame").position_world
control_frames["right_hand"]=range(30,60)
control_hints["right_hand"]=parse_query_obj("fruit").position_world
pelvis_height=0.9
specify_joint_position(control_text, control_joints, control_frames, control_hints,pelvis_height,n_frames)
generate_motion(control_text, n_frames)

# Query: A person eats the fruit he is holding.
control_text='A person eats the fruit he is holding.'
n_frames=180
control_frames=[]
control_hints=[]
control_frames=range(0,180)
control_hints=parse_query_obj("last created motion frame").position_world
pelvis_height=0.9
specify_pelvis_position(control_text, control_frames, control_hints,pelvis_height, n_frames)
generate_motion(control_text, n_frames)

# Query: A person walks to the sofa.
control_text='A person walks.'
n_frames=180
target=parse_query_obj('sofa')
affordance_map=get_affordance_map(f'a point at {target.position_voxel}')
avoidance_map=get_avoidance_map(f'anything')
specify_pelvis_trajectory(affordance_map,avoidance_map,control_text,n_frames)
generate_motion(control_text, n_frames)

# Query: A person lies on the sofa.
control_text='A person lies on the sofa.'
n_frames=120
control_frames=[]
control_hints=[]
control_frames=range(60,120)
control_hints=parse_query_obj("last created motion frame").position_world
pelvis_height=0.4
specify_pelvis_position(control_text, control_frames, control_hints,pelvis_height, n_frames)
generate_motion(control_text, n_frames)

# Query: A person falls asleep quickly.
control_text='A person falls asleep quickly.'
n_frames=180
control_frames=[]
control_hints=[]
control_frames=range(0,180)
control_hints=parse_query_obj("last created motion frame").position_world
pelvis_height=0.4
specify_pelvis_position(control_text, control_frames, control_hints,pelvis_height, n_frames)
generate_motion(control_text, n_frames)