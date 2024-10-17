
# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# _functions for fixing random seed
from utils.fixseed import fixseed
import os
import numpy as np
import torch
# _parse command line arguments
from utils.parser_util import generate_args
# -used to create CMDM and SpacedDiffusion
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.rotation_conversions import quaternion_to_matrix
from utils import dist_util
# -used for distributed training
from model.cfg_sampler import ClassifierFreeSampleModel
# -used for load dataset
from data_loaders.get_data import get_dataset_loader
# -recover joints' world coordinates from Root in Coordinate
from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_root_rot_pos
# -defines constant parameters such as kinematic chain and raw offsets
import data_loaders.humanml.utils.paramUtil as paramUtil
# -generate stick animations from generated joint data
from data_loaders.humanml.utils.plot_script import plot_3d_motion
# -library for file operations
import shutil
# -parse data in a batch, generate motion and condition from input batches
from data_loaders.tensors import collate
# -generate and parse control information
# from utils.text_control_example import collate_all
# from utils.my_control import collate_all
# -join two paths
from os.path import join as pjoin
from utils.custom import get_rigid_transform, local2global, global2local

args = generate_args()
fixseed(args.seed)
# -basename get the file name in the end of the path
name = os.path.basename(os.path.dirname(args.model_path))
# -niter=number iteration
niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
fps = 12.5 if args.dataset == 'kit' else 20
# -if no text prompt assigned, use data in dataset instead
is_using_data = not any([args.text_prompt])

dist_util.setup_dist(args.device)
# rotation and translation matrixes that transfers a local position in one motion sequence to a global one

first_run=True
R=[]
t=[]

def generate_motion(n_frames, out_path, texts, hints,index,epoch,base_position=None):

    
    global first_run
    if base_position is not None:
        eye=np.eye(3)
        inner_R=[]
        for i in range(len(base_position)):
            inner_R.append(eye)
        inner_t=base_position
        R.append(inner_R)
        t.append(inner_t)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        if args.text_prompt == 'predefined':
            args.num_samples = len(texts)
            if args.cond_mode == 'only_spatial':
                # only with spatial control signal, and the spatial control signal is defined in utils/text_control_example.py
                texts = ['' for i in texts]
            elif args.cond_mode == 'only_text':
                # only with text prompt, and the text prompt is defined in utils/text_control_example.py
                hints = None
        else:
            # otherwise we use text_prompt
            texts = [args.text_prompt]
            args.num_samples = 1
            hint = None
    if np.all(hints == 0):
        print("Generate is text only because hints is all zeros.")
        hints=None
        args.cond_mode = 'only_text'

    # assert args.num_samples <= args.batch_size, \
    #     f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    # load dataset only for the first run
    global data,total_num_samples,model,diffusion,state_dict
    if first_run:
        # first_run=False
        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
        total_num_samples = args.num_samples * args.num_repetitions

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking



    # # load previously generated samples
    # uncomment this and below for GCML generation
    # last_raw_sample_path=f"{out_path[:-6]}{epoch-1}/raw_sample.npy"
    # last_xyz_sample_path=f"{out_path[:-6]}{epoch-1}/xyz_sample.npy"
    # load previously generated samples
    last_raw_sample_path=f"{out_path[:-1]}{index-1}/raw_sample.npy"
    last_xyz_sample_path=f"{out_path[:-1]}{index-1}/xyz_sample.npy"

    # caculate rotation and translation matrix R and t
    if os.path.exists(last_xyz_sample_path):
        loaded_motions=np.load(last_xyz_sample_path)        #[3,22,3,120]
        # loaded_motions=loaded_motions[index]
        loaded_motions=loaded_motions

        loaded_raw_sample=np.load(last_raw_sample_path)
        # loaded_raw_sample=loaded_raw_sample[index]
        loaded_raw_sample=loaded_raw_sample
        loaded_raw_sample=torch.from_numpy(loaded_raw_sample).float()
        loaded_raw_sample = loaded_raw_sample.permute(0, 3, 2, 1).contiguous()
        loaded_raw_sample = loaded_raw_sample.squeeze(2)
        data_root = './dataset/HumanML3D'
        mean = torch.from_numpy(np.load(pjoin(data_root, 'Mean.npy'))).float()
        std = torch.from_numpy(np.load(pjoin(data_root, 'Std.npy'))).float()
            
        loaded_raw_sample = loaded_raw_sample * std + mean
        R_,t_ = recover_root_rot_pos(loaded_raw_sample)
        inner_R=[]
        inner_t=[]
        for i in range(loaded_motions.shape[0]):
            recover_R=quaternion_to_matrix(R_[i,-1,:]).numpy()
            # we dont know why, but the matrix caculated this way should be fixed somewhat to maintain coherence
            recover_R[0,2]=-recover_R[0,2]
            recover_R[2,0]=-recover_R[2,0]
            recover_t=t_[i,-1,:].numpy()
            recover_t[1]=0
            inner_R.append(recover_R)
            inner_t.append(recover_t)
            # concatenate with previous R and t for contigous generation
        R.append(inner_R)
        t.append(inner_t)

    # transform global hints to local
    if len(R)!=0 and hints is not None:
        hints=hints.transpose(0,2,3,1)        #[3,22,3,196]
        original_hints = hints.copy()
        for i in range(hints.shape[0]):
            for j in range(len(R)):
                hints[i,:,:,:] = global2local(hints[i,:,:,:], R[j][i], t[j][i])
        hints[original_hints == 0] = 0
        hints=hints.transpose(0,3,1,2)
    if hints is not None:
        hints=hints.reshape(hints.shape[0],n_frames,66)


    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        # -generate model_keywordarguments from texts and hints
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        # t2m
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        if hints is not None:
            collate_args = [dict(arg, hint=hint) for arg, hint in zip(collate_args, hints)]

        _, model_kwargs = collate(collate_args)
    
    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())
    # use MDM's origional inbetweening method, replace the first frame 
    # in new generation with the last frame in previous generated motion
    if os.path.exists(last_raw_sample_path):
        print(f"loaded last motion frame from {last_raw_sample_path}")

        loaded_motions=np.load(last_raw_sample_path)
        # loaded_motions=loaded_motions[index]
        loaded_motions=loaded_motions
        motion_count=loaded_motions.shape[0]

        # make loaded motions the same shape as next generating motion
        input_motions=np.zeros((motion_count, 263, 1, n_frames), dtype=np.float32)
        impainted_frames=5
        input_motions[..., 0] = loaded_motions[...,-1]

        input_motions=torch.from_numpy(input_motions)
        input_motions=input_motions.to(dist_util.dev())
        model_kwargs['y']['inpainted_motion'] = input_motions
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                                device=input_motions.device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, 1:] = False


    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []  
    all_hint_for_vis = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop
        # -here is where samples are generated
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        raw_sample=sample.cpu().numpy()

        sample = sample[:, :263]    #[3,263,3,120]
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()   #[3,1,120,263]
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)     #[3,22,3,120]
        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep    
        #-xyz in default, so the following line is usually not in use
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        xyz_sample=sample.cpu().numpy()
        # move subsequent motion globally to align with the last frame of previous motion

        if len(R)!=0:
            for i in range(xyz_sample.shape[0]):
                for j in range(len(R)):
                    xyz_sample[i,:,:,:] = local2global(xyz_sample[i,:,:,:], R[-j-1][i], t[-j-1][i])
            if 'hint' in model_kwargs['y']:
                hint = model_kwargs['y']['hint'].cpu()    #[3,196,66]
                hint=hint.reshape(hint.shape[0],hint.shape[1],22,3)
                hint=hint.numpy()
                hint=hint.transpose(0,2,3,1)        #[3,22,3,196]
                original_hint = hint.copy()
                for i in range(hint.shape[0]):
                    for j in range(len(R)):
                        hint[i,:,:,:] = local2global(hint[i,:,:,:], R[-j-1][i], t[-j-1][i])
                hint[original_hint == 0] = 0
                hint=hint.transpose(0,3,1,2).reshape(hint.shape[0],n_frames,66)
                model_kwargs['y']['hint']=torch.from_numpy(hint).to(dist_util.dev())
            sample=torch.from_numpy(xyz_sample)
 

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]
            # -parse hints and add them in all_hint and all_hint_for_vis
            if 'hint' in model_kwargs['y']:
                hint = model_kwargs['y']['hint']
                # denormalize hint
                if args.dataset == 'humanml':
                    spatial_norm_path = './dataset/humanml_spatial_norm'
                elif args.dataset == 'kit':
                    spatial_norm_path = './dataset/kit_spatial_norm'
                else:
                    raise NotImplementedError('unknown dataset')
                raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))).cuda(args.device)
                raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))).cuda(args.device)
                mask = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(-1) != 0
                # -cancel all normalizing steps to hints
                # hint = hint * raw_std + raw_mean
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask.unsqueeze(-1)
                hint = hint.view(hint.shape[0], hint.shape[1], -1)
                # ---
                all_hint.append(hint.data.cpu().numpy())
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3)
                all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    # -
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if 'hint' in model_kwargs['y']:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    # -caculate distance between generated motion and designed route
    if len(all_hint) != 0:
        from utils.simple_eval import simple_eval
        results = simple_eval(all_motions, all_hint, n_joints)
        print(results)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths, "hint": all_hint_for_vis,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))
    raw_sample_path=os.path.join(out_path, 'raw_sample.npy')
    xyz_sample_path=os.path.join(out_path, 'xyz_sample.npy')

    np.save(raw_sample_path, raw_sample)
    np.save(xyz_sample_path, xyz_sample)

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7
    # -generate file name template, including :02d to be replaced
    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    # -plot visualize animation
    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            if 'hint' in model_kwargs['y']:
                hint = all_hint_for_vis[rep_i*args.batch_size + sample_i]
            else:
                hint = None
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')
    return all_motions

# -generate ffmpeg commands and excute them in commandline to generate animations
def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]

    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files

# -generate filename template
def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data
