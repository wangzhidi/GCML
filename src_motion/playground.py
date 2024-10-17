import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import open3d as o3d

openai.api_key = "sk-WvkipvKc2xzF6DhG6c1303A6087e44Ff9a246aB62d065cA6"  # set your API key here
openai.api_base = 'https://api.pumpkinaigc.online/v1'

config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
for lmp_name, cfg in config['lmp_config']['lmps'].items():
    cfg['model'] = 'gpt-4-1106-preview'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])


condition_path="/home/wangd/afford-motion-main/sample.npy"
conditions=np.load(condition_path, allow_pickle=True)

for i in range(0,64):
    print(f"______________________________________running iteration {i}_____________________________________________")
    ply_path={}
    item=conditions[i]
    text=item["c_text"]
    ply_path['path']=item["info_scene_mesh"]
    ply_path['trans']=item["info_scene_trans"]
    ply_path['index']=i

    
    lmps, lmp_env = setup_LMP(ply_path, visualizer, config, debug=False, env_name="omnicontrol")
    voxposer_ui = lmps['plan_ui']
    # object_names will not be used currently
    object_names=[]
    set_lmp_objects(lmps, object_names)  # set the object names to be used by voxposer

    instruction = text
    voxposer_ui(instruction)





