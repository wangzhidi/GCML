import os
import random
import numpy as np
import logging
import argparse
import urllib

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from openscene.util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from openscene.util import config
from openscene.util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from openscene.run.distill import get_model

from openscene.dataset.label_constants import *
from openscene.run.visualize import visualize
import open3d as o3d
import tensorflow as tf

def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(positive, negative):
    labelset=list((positive,)+negative)
    palette=get_palette(num_cls=len(labelset),colormap="none")


    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')


    return text_features, labelset, palette

def init():
    '''Main function.'''

    # args = get_parser()
    args=np.load("/home/wangd/GCML/args.npy",allow_pickle=True)
    args=args[None][0]
    args['feature_type']='ensemble'
    # np.save("args.npy", args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    
    # By default we do not use shared memory for evaluation
    if not hasattr(args, 'use_shm'):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        return(main_worker(args.test_gpu, args.ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        # logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == 'fusion':
        pass # do not need to load weight
    elif is_url(args.model_path): # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            # logger.info('Loaded a parallel model')

        # if main_process():
        #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    
    from openscene.dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset

    return(evaluate(model, val_loader, labelset_name))

def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy: # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, 'vis_input') and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, 'vis_pred') and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, 'vis_gt') and args.vis_gt:
        vis_gt = True



    with torch.no_grad():
        model.eval()
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            # if main_process():
            #     logger.info(
            #         "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            if rep_i>0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []
            return model, feature_type , val_data_loader
def find(model, feature_type, val_data_loader, path,trans, positive, negative):
    text_features, labelset, palette = \
        precompute_text_related_properties(positive,negative)
    val_data_loader.dataset.path=path
    val_data_loader.dataset.trans=trans
    for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
        sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        coords = coords[inds_reverse, :]
        pcl = coords[:, 1:].cpu().numpy()

        if feature_type == 'distill':
            predictions = model(sinput)
            predictions = predictions[inds_reverse, :]
            pred = predictions.half() @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].cpu()
        elif feature_type == 'fusion':
            predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
            pred = predictions.half() @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].detach().cpu()

        elif feature_type == 'ensemble':
            feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
            # pred_fusion = feat_fuse.half() @ text_features.t()
            pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

            predictions = model(sinput)
            predictions = predictions[inds_reverse, :]
            # pred_distill = predictions.half() @ text_features.t()
            pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

            # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
            # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
            # pred = pred_distill
            # pred[mask_ensem] = pred_fusion[mask_ensem]
            # logits_pred = torch.max(pred, 1)[1].detach().cpu()

            feat_ensemble = predictions.clone().half()
            mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
            feat_ensemble[mask_] = feat_fuse[mask_]
            pred = feat_ensemble @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].detach().cpu()

            predictions = feat_ensemble # if we need to save the features
        else:
            raise NotImplementedError
        

        pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)# visualize(pcl,(input_color+1)/2,logits_pred)
        origional_pcl=val_data_loader.dataset.origional_pcl
        origional_color=val_data_loader.dataset.origional_color
        # center=visualize(origional_pcl,pred_label_color,logits_pred,labelset)
        center=visualize(origional_pcl,origional_color,logits_pred,labelset)
        return center



