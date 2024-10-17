## VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models



<img  src="static\images\fig1.jpg" width="550">

This is the  code for GCML: Grounding Complex Motions using Large Language Model in 3D Scenes.

## Abstract
To solve the problem of generating complex motions, we introduce GCML (Grounding Complex Motions using a Large Language Model). This method supports complex texts and scenes as inputs, such as mopping the floor in a cluttered room. Such everyday actions are challenging for current motion generation models for two main reasons. First, such complex actions are rarely found in existing HSI datasets, which places high demands on the generalization capabilities of current data-driven models. Second, these actions are composed of multiple stages, with considerable variation, making it difficult for models to understand and generate the appropriate motions. Current methods in the HSI field can control the generation of simple actions under multiple constraints, such as walking joyfully toward a door, but they cannot handle the complexity of tasks like the one described above. By incorporating a Large Language Model and a 3D Visual Grounding Model into the HSI domain, our approach can decompose a complex user prompt into a sequence of simpler subtasks and identify interaction targets and obstacles within the scene. Based on these subtask descriptions and spatial control information, the Motion Generation Model generates a sequence of full-body motions, which are then combined into a long motion sequence that aligns with both the user's input and the scene semantics. Experimental results demonstrate that our method achieves competitive performance for simple action generation on the HUMANISE dataset and the generalization evaluation set. For complex motion generation, we created a new evaluation set by automatically generating possible behaviors of virtual humans in common indoor scenes, where our method significantly outperforms existing approaches.

## Setup

# Step 1: Install MinkowskiEngine
Follow the steps in its official page: https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#installation 

# Step 2: Prepare Openscene Datasets
Start by cloning the repo:
```bash
git clone --recursive git@github.com:pengsongyou/openscene.git
cd openscene
```
# Step 3: Prepare Omnicontrol Datasets
Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
- Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

# Step 3: Install other dependencies
- Create a conda environment:
```Shell
conda create -n gcml python=3.9
conda activate gcml
pip install -r requirements.txt
```
- Obtain an OpenAI API key, and put it in playground file.


## Setup

- See [Instructions](https://github.com/stepjam/RLBench#install) to install PyRep and RLBench (Note: install these inside the created conda environment).

- Install other dependencies:
```Shell
pip install -r requirements.txt
```

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key, and put it inside the first cell of the demo notebook.

## Running Demo

Demo code is at src_motion/playground_base.py. Instructions can be found in the notebook.

