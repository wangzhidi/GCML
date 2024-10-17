from LMP import LMP
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import transforms3d
from scipy.interpolate import interp1d
from openscene.run.evaluate import init, find
import open3d as o3d

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

class LMP_interface():

  def __init__(self, ply_path, lmp_config, controller_config, planner_config, visualizer=None):
    self.path=ply_path['path']
    self.trans=ply_path['trans']
    self.index=ply_path['index']
    self.env_mesh = o3d.io.read_triangle_mesh(self.path)

    self.env_mesh.transform(self.trans)

    self.vertices=np.asarray(self.env_mesh.vertices)
    self.workspace_bounds_max=self.vertices.max(axis=0)
    self.workspace_bounds_min=self.vertices.min(axis=0)
    self.visualizer=visualizer
    self.obstacle_bounding_box=[]
    self.obstacle_bounding_box_world=[]
    self.start_position=[0,0,0.5]
    self.base_position=[self.start_position[0],self.start_position[1],0]  #base_position indicates the first frame of human, do not change
    
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self._planner = PathPlanner(planner_config, map_size=self._map_size)
    self.control_info={}
    self.control_info["n_frames"]=[]
    self.control_info["texts"]=[]
    self.control_info["hints"]=[]
    self.control_info["joint_ids"]=[]
    self.control_info["base_position"]=[]
    self.control_info["env_trans"]=[]
    self.n_frame=0
    self.text=""
    self.hint=[]
    self.joint_id=[]
    self.scene_pcd = self.env_mesh.sample_points_uniformly(number_of_points=len(self.env_mesh.vertices))
    self.scene_points=np.asarray(self.scene_pcd.points)
    self.scene_points_colors=np.asarray(self.scene_pcd.colors)
    self.model, self.feature_type , self.val_data_loader=init()
    

    # calculate size of each voxel (resolution)
    self._resolution = (self.vertices.max(axis=0) - self.vertices.min(axis=0)) / self._map_size
    print('#' * 50)
    print(f'## voxel resolution: {self._resolution}')
    print('#' * 50)
    print()
    print()
  
  # ======================================================
  # == functions exposed to LLM
  # ======================================================


  
  
  def detect(self, positive, negative):
    """return an observation dict containing useful information about the object, currently fixed"""
    obs_dict = dict()


    # center, aabb=find(self.model,self.feature_type,self.val_data_loader,self.path,self.trans, positive,negative)
    # if center is None:
    #   #just assign the base value to prevent program termination
    #   obs_dict['name'] = positive
    #   obs_dict['position_voxel'] = self._world_to_voxel(np.array([0,0,1]))  # in voxel frame
    #   obs_dict['aabb'] = np.array(np.array([[0,0,0],[1,1,1]]))  # in voxel frame
    #   obs_dict['position_world'] = [0,0,1]  # in world frame
    # else:
    #   obs_dict['name'] = positive
    #   obs_dict['position_voxel'] = self._world_to_voxel(center)  # in voxel frame
    #   obs_dict['aabb'] = np.array(aabb)  # in voxel frame
    #   obs_dict['position_world'] = center  # in world frame
    self.start_position=[0.5,1.5,0.5]
    obs_dict['name'] = positive
    obs_dict['position_voxel'] = self._world_to_voxel(np.array([4.5,7.0,0.5]))  # in voxel frame
    obs_dict['aabb'] = np.array(np.array([[0,0,0],[1,1,1]]))  # in voxel frame
    obs_dict['position_world'] = [0,0,1]  # in world frame

    object_obs = Observation(obs_dict)
    return object_obs
  
  def specify_pelvis_trajectory(self, affordance_map=None, avoidance_map=None,control_text='',n_frames=120):
    """
    First use planner to generate waypoint path, then use controller to follow the waypoints.

    Args:
      movable_obs_func: callable function to get observation of the body to be moved
      affordance_map: callable function that generates a 3D numpy array, the target voxel map
      avoidance_map: callable function that generates a 3D numpy array, the obstacle voxel map
      rotation_map: callable function that generates a 4D numpy array, the rotation voxel map (rotation is represented by a quaternion *in world frame*)
      velocity_map: callable function that generates a 3D numpy array, the velocity voxel map
      gripper_map: callable function that generates a 3D numpy array, the gripper voxel map
    """
    # initialize default voxel maps if not specified
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle')
    execute_info = []
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(self._cfg['max_plan_iter']):
        step_info = dict()
        # evaluate voxel maps such that we use latest information
        _affordance_map = affordance_map()
        _avoidance_map = avoidance_map
        # preprocess avoidance map
        # _avoidance_map = self._preprocess_avoidance_map(_avoidance_map, _affordance_map, movable_obs)
        # start planning
        start_pos = self._world_to_voxel(np.array([self.start_position[0], self.start_position[1], 0.5]))
        start_time = time.time()
        # optimize path and log
        path_voxel, planner_info = self._planner.optimize(start_pos, _affordance_map, _avoidance_map)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'
        path_world=[]
        for voxel in path_voxel:
          voxel_world=self._voxel_to_world(voxel)
          path_world.append(voxel_world)
        for boundingbox in self.obstacle_bounding_box:
          self.obstacle_bounding_box_world.append(self._voxel_to_world(boundingbox))

        step_info['path_voxel'] = path_voxel
        step_info['planner_info'] = planner_info
        step_info['start_pos'] = start_pos
        step_info['plan_iter'] = plan_iter
        step_info['affordance_map'] = _affordance_map
        step_info['avoidance_map'] = _avoidance_map
        step_info['control_text']=control_text
        step_info['path_world']=path_world
        step_info['env_mesh']=self.env_mesh
        step_info['vertices']=self.vertices
        step_info['scene_pcd']=self.scene_pcd
        step_info['scene_points']=self.scene_points
        step_info['scene_points_colors']=self.scene_points_colors
        step_info['workspace_bounds_max']=self.workspace_bounds_max
        step_info['workspace_bounds_min']=self.workspace_bounds_min
        step_info['obstacle_bounding_box']=self.obstacle_bounding_box
        step_info['obstacle_bounding_box_world']=self.obstacle_bounding_box_world
        self.text=control_text
        self.n_frame=n_frames
        hint = np.zeros((n_frames, 3))
        # reshape generated path as the same shape of hint
        path_world_np=np.asarray(path_world)
        indices = np.linspace(0, n_frames-1, len(path_world), dtype=int)
        hint[indices]=path_world_np
        # interpolate zero elements
        mask=np.any(hint!=0,axis=1)
        indices=np.arange(hint.shape[0])
        for i in range(hint.shape[1]):
          non_zero_indices=indices[mask]
          non_zero_values=hint[mask,i]
          interp_func = interp1d(non_zero_indices, non_zero_values, kind='linear', bounds_error=False, fill_value="extrapolate")
          hint[:, i] = interp_func(indices)
        # make z axis constant: 0.9 meters high
        hint[:,2]=0.9
        self.joint_id=[0]
        self.start_position=path_world[-1]
        self.hint=np.zeros((1,n_frames,22,3))
        self.hint[:,:,0,:]=hint


        # print(f'obstacle voxels summed:{np.sum(_avoidance_map==1)}')

        # visualize
        if self._cfg['visualize']:
          assert self.visualizer is not None
          step_info['start_pos_world'] = self._voxel_to_world(start_pos)
          step_info['targets_world'] = self._voxel_to_world(planner_info['targets_voxel'])
          self.visualizer.visualize(step_info)

        # execute path
        print("__________SPECIFY PELVIS TRAJECTORY_____________")

    return step_info
  
  def specify_joint_position(self,control_text, control_joints, control_frames, control_hints, pelvis_height=0.9, n_frames=120):
    """
      generate texts and hints that specify hand position at certain frames,
      in such an implementation, only one position can be specified for a joint at one frame
    """
    self.joint_id=[]
    self.hint=np.zeros((1,n_frames,22,3))
    for joint in control_joints:
      if joint=="pelvis":
        joint_id=0
      if joint=="left_foot":
        joint_id=10
      if joint=="right_foot":
        joint_id=11
      if joint=="head":
        joint_id=15
      if joint=="left_hand":
        joint_id=20
      if joint=="right_hand":
        joint_id=21
      self.joint_id.append(joint_id)
      for frame in control_frames[joint]:
        self.hint[:,frame,joint_id,:]=control_hints[joint]
        if joint=="pelvis":
          self.hint[:,frame,joint_id,2]=pelvis_height
      if joint=="pelvis":
          self.start_position=self.hint[0,-1,0,:]

    self.text=control_text
    self.n_frame=n_frames

  def specify_pelvis_position(self,control_text, control_frames, control_hints, pelvis_height, n_frames=120):
    """
      generate texts and hints that specify hand position at certain frames,
      in such an implementation, only one position can be specified for a joint at one frame
    """
    self.joint_id=[0]
    self.hint=np.zeros((1,n_frames,22,3))
    for frame in control_frames:
      self.hint[:,frame,0,:]=control_hints
      self.hint[:,frame,0,2]=pelvis_height

    self.text=control_text
    self.n_frame=n_frames
    self.start_position=self.hint[0,-1,0,:]


    # execute path
    print("__________SPECIFY JOINT POSITION_____________")
  def specify_nothing(self, control_text, n_frames=120):
    """
      generate texts but no hints
    """
    self.text=control_text
    self.n_frame=n_frames
    self.hint=np.zeros((1,n_frames,22,3))
    self.joint_id=[]
    print("__________SPECIFY NOTHING_____________")


  def generate_motion(self, control_text, n_frames=120):
    """
      generate texts and hints that specify hand position at certain frames
    """
    self.text=control_text
    self.n_frame=n_frames
    self.control_info["n_frames"].append(self.n_frame)
    self.control_info["texts"].append(self.text)
    self.control_info["hints"].append(self.hint)
    self.control_info["joint_ids"].append(self.joint_id)
    self.control_info["base_position"].append(self.base_position)
    self.control_info["env_trans"].append(self.trans)
    file_count = 0


    np.save(f"output/custom/control_info{self.index}.npy", self.control_info)

    # execute path

  
  def cm2index(self, cm, direction):
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)

  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
      aabb=np.array([[min_x,min_y,min_z], [max_x,max_y,max_z]])  # in voxel frame
      self.obstacle_bounding_box.append(aabb)

    return voxel_map
  
  def set_voxel_by_bounding_box(self, voxel_map, aabb, value=1):
    """given a bounding box in voxel map from detect function, set the value of the voxels in the bounding box."""
    min_x = max(0, aabb[0][0])
    max_x = min(self._map_size, aabb[1][0])
    min_y = max(0, aabb[0][1])
    max_y = min(self._map_size, aabb[1][1])
    min_z = max(0, aabb[0][2])
    max_z = min(self._map_size, aabb[1][2])
    voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    print(f'obstacle voxels summed:{np.sum(voxel_map==1)}')
    self.obstacle_bounding_box.append(aabb)
    return voxel_map
  
  def get_empty_affordance_map(self):
    return self._get_default_voxel_map('target')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

  def get_empty_avoidance_map(self):
    return self._get_default_voxel_map('obstacle')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_rotation_map(self):
    return self._get_default_voxel_map('rotation')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_velocity_map(self):
    return self._get_default_voxel_map('velocity')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_gripper_map(self):
    return self._get_default_voxel_map('gripper')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  
  # ======================================================
  # == helper functions
  # ======================================================
  def _world_to_voxel(self, world_xyz):
    """transforms a world coordinate to voxel"""
    _world_xyz = world_xyz.astype(np.float32)
    voxel_xyz = pc2voxel(_world_xyz, self.workspace_bounds_min, self.workspace_bounds_max, self._map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    world_xyz = voxel2pc(voxel_xyz, self.workspace_bounds_min, self.workspace_bounds_max, self._map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center


  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  
  def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
    """
    convert path (generated by planner) to trajectory (used by controller)
    path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
    """
    # convert path to trajectory
    traj = []
    for i in range(len(path)):
      # get the current voxel position
      voxel_xyz = path[i]
      # get the current world position
      world_xyz = self._voxel_to_world(voxel_xyz)
      voxel_xyz = np.round(voxel_xyz).astype(int)
      # get the current rotation (in world frame)
      rotation = rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current velocity
      velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current on/off
      gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
      if (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
        # get indices of the less common values
        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
        less_common_indices = np.where(gripper_map == less_common_value)
        less_common_indices = np.array(less_common_indices).T
        # get closest distance from voxel_xyz to any of the indices that have less common value
        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
        # if the closest distance is less than threshold, then set gripper to less common value
        if closest_distance <= 3:
          gripper = less_common_value
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}')
      # add to trajectory
      traj.append((world_xyz, rotation, velocity, gripper))
    # append the last waypoint a few more times for the robot to stabilize
    for _ in range(2):
      traj.append((world_xyz, rotation, velocity, gripper))
    return traj
  
  def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    scene_collision_map = self._get_scene_collision_voxel_map()
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    try:
      ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
      scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    except KeyError:
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1)
    return avoidance_map

  def preprocess_avoidance_map(self):
    avoidance_map=self.get_empty_avoidance_map()

    voxel_points=self._points_to_voxel_map(self.scene_points)
    avoidance_map=voxel_points
    return avoidance_map


def setup_LMP(ply_path,visualizer, general_config, debug=False,env_name="omnicontrol"):
  controller_config = general_config['controller']
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']
  lmps_config = general_config['lmp_config']['lmps']
  # LMP env wrapper
  lmp_env = LMP_interface(ply_path, lmp_env_config, controller_config, planner_config, visualizer)
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'euler2quat': transforms3d.euler.euler2quat,
      'quat2euler': transforms3d.euler.quat2euler,
      'qinverse': transforms3d.quaternions.qinverse,
      'qmult': transforms3d.quaternions.qmult,
  }  # external library APIs
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }  # our custom APIs exposed to LMPs

  # allow LMPs to access other LMPs
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config']]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)

  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer
  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name
  )

  lmps = {
      'plan_ui': task_planner,
      'composer_ui': composer,
  }
  lmps.update(low_level_lmps)

  return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_min, voxel_bounds_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_min, voxel_bounds_max)
  # voxelize
  voxels = (pc - voxel_bounds_min) / (voxel_bounds_max - voxel_bounds_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  voxels = np.array(voxels)
  voxels = np.where(voxels < 1, 1, voxels)  
  voxels = np.where(voxels > map_size-1, map_size-1, voxels)  
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  # for i in range(points_vox.shape[0]):
  #     voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  radius=1
  # set all voxels within radius as 1
  for i in range(points_vox.shape[0]):
      x, y, z = points_vox[i]
      x_min = max(x - radius, 0)
      x_max = min(x + radius, map_size - 1)
      y_min = max(y - radius, 0)
      y_max = min(y + radius, map_size - 1)
      z_min = max(z - radius, 0)
      z_max = min(z + radius, map_size - 1)
      voxel_map[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
  return voxel_map