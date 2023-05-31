import sys, os
import yaml
import argparse

from pathlib import Path
import h5py
from skimage import io
import numpy as np
import time
import warnings

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

import torch

from stardist_tools import calculate_extents, Rays_GoldenSpiral
from stardist_tools.matching import matching, matching_dataset
from stardist_tools.csbdeep_utils import download_and_extract_zip_file


from src.training import train
from src.data.stardist_dataset import get_train_val_dataloaders
from src.utils import seed_all, prepare_conf, plot_img_label
from src.models.config import ConfigBase, Config3D
from src.models.stardist3d import StarDist3D


def main():
    parser = argparse.ArgumentParser(description='Train StarDist3D model')
    parser.add_argument('--config', type=str, help='path to configuration file', required=True)

    # Load YAML file
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert str(config['Threshold_optimizer']['dataset']) in ['train', 'val'], "No valid dataset has been provided for threshold optimization. Please determine whether to use the training ('train') or validation ('val') dataset for NMS threshold optimization."
    assert str(config['Threshold_optimizer']['epoch']) in ['last', 'best'], "No valid epoch has been provided for threshold optimization. Please determine whether to use the 'best' or 'last' epoch for NMS threshold optimization."

    print(config)

    conf = Config3D(
        name                           = config['Attributes']['name'],
        random_seed                    = config['Attributes']['random_seed'],
        
        # ========================= Networks configurations ==================
        init_type                       = config['Networks']['init_type'],
        init_gain                       = config['Networks']['init_gain'],

        backbone                        = config['Networks']['backbone'],
        grid                            = config['Networks']['grid'],
        anisotropy                      = config['Networks']['anisotropy'],

        n_channel_in                   = config['Networks']['n_channel_in'],
        kernel_size                    = config['Networks']['kernel_size'],
        resnet_n_blocks                = config['Networks']['resnet_n_blocks'],
        resnet_n_downs                 = config['Networks']['resnet_n_downs'],
        n_filter_of_conv_after_resnet  = config['Networks']['n_filter_of_conv_after_resnet'],
        resnet_n_filter_base           = config['Networks']['resnet_n_filter_base'],
        resnet_n_conv_per_block        = config['Networks']['resnet_n_conv_per_block'],
        use_batch_norm                  = config['Networks']['use_batch_norm'],

        #======================================================================

        # ========================= dataset ==================================
        data_dir                       = r"{}".format(config['Dataset']['data_dir']),
        val_size                       = config['Dataset']['val_size'],
        
        n_rays                         = config['Dataset']['Parameters']['n_rays'],

        foreground_prob                = config['Dataset']['Parameters']['foreground_prob'],
        n_classes                      = None,
        patch_size                     = config['Dataset']['Parameters']['patch_size'],
        cache_sample_ind               = config['Dataset']['Parameters']['cache_sample_ind'],
        cache_data                     = config['Dataset']['Parameters']['cache_data'],

        batch_size                     = config['Dataset']['batch_size'],
        num_workers                    = config['Dataset']['num_workers'],

        preprocess                     = config['Dataset']['preprocessing']['preprocess'],
        preprocess_val                 = config['Dataset']['preprocessing']['preprocess_val'],
        intensity_factor_range         = config['Dataset']['preprocessing']['intensity_factor_range'],
        intensity_bias_range           = config['Dataset']['preprocessing']['intensity_bias_range'],

        #======================================================================


        # ========================= Training ==================================

        use_gpu                        = True if torch.cuda.is_available()and config['Trainer']['use_gpu'] else False,
        #gpu_ids                       = config['Trainer']['gpu_ids'],
        use_amp                        = config['Trainer']['use_amp'],
        isTrain                        = config['Trainer']['isTrain'] ,
        evaluate                       = config['Trainer']['evaluate'],

        
        load_epoch                     = config['Trainer']['load_epoch'],
        n_epochs                       = config['Trainer']['n_epochs'],
        n_steps_per_epoch              = config['Trainer']['n_steps_per_epoch'],

        save_epoch_freq                = config['Trainer']['save_epoch_freq'],
        start_saving_best_after_epoch  = config['Trainer']['start_saving_best_after_epoch'],

        lambda_prob                    = config['Trainer']['Parameters']['lambda_prob'],
        lambda_dist                    = config['Trainer']['Parameters']['lambda_dist'],
        lambda_reg                     = config['Trainer']['Parameters']['lambda_reg'],
        lambda_prob_class              = config['Trainer']['Parameters']['lambda_prob_class'],

        #======================================================================


        # ========================= Optimizers ================================
        lr                             = config['Optimizer']['lr'],
        beta1                          = config['Optimizer']['beta1'],
        beta2                          = config['Optimizer']['beta2'],

        lr_policy                      = config['Optimizer']['lr_policy'],
        lr_plateau_factor              = config['Optimizer']['lr_plateau_factor'],
        lr_plateau_threshold           = config['Optimizer']['lr_plateau_threshold'],
        lr_plateau_patience            = config['Optimizer']['lr_plateau_patience'],
        min_lr                         = config['Optimizer']['min_lr'],
        
        lr_linear_n_epochs             = config['Optimizer']['lr_linear_n_epochs'],
        lr_decay_iters                 = config['Optimizer']['lr_decay_iters'],
        T_max                          = config['Optimizer']['T_max'])


    conf.checkpoints_dir = config['Attributes']['output_dir'] + '/checkpoints'
    conf.log_dir = config['Attributes']['output_dir'] + '/logs'
    conf.result_dir = config['Attributes']['output_dir'] + '/results'

    seed_all(conf.random_seed)

    opt = prepare_conf(conf)

    model = StarDist3D(opt)

    print(model)

    print("Total number of epochs".ljust(25), ":", model.opt.n_epochs)

    fov = np.array( [max(r) for r in model._compute_receptive_field()] )
    object_median_size = opt.extents

    print("Median object size".ljust(25), ":", object_median_size)
    print("Network field of view".ljust(25), ":", fov)

    if any(object_median_size > fov):
        warnings.warn("WARNING: median object size larger than field of view of the neural network.")

    rays = Rays_GoldenSpiral(opt.n_rays, anisotropy=opt.anisotropy)

    train_dataloader, val_dataloader = get_train_val_dataloaders(opt, rays)

    total_nb_samples = len( train_dataloader.dataset ) + ( len(val_dataloader.dataset) if val_dataloader is not None else 0 )
    nb_samples_train = len(train_dataloader.dataset)
    nb_samples_val = total_nb_samples - nb_samples_train

    print("Total nb samples: ".ljust(40), total_nb_samples)
    print("Train nb samples: ".ljust(40), nb_samples_train)
    print("Val nb samples: ".ljust(40), nb_samples_val)

    print("Train augmentation".ljust(25), ":",  train_dataloader.dataset.opt.preprocess)
    print("Val augmentation".ljust(25), ":", val_dataloader.dataset.opt.preprocess)

    train(model, train_dataloader, val_dataloader)

    if str(config['Threshold_optimizer']['dataset']) == 'train':
        print("Using training dataset for NMS threshold optimization:")
        X, Y = train_dataloader.dataset.get_all_data()
        conf.load_epoch = conf.n_epochs
        model = StarDist3D(conf)

    if str(config['Threshold_optimizer']['dataset']) == 'val':
        print("Using validation dataset for NMS threshold optimization:")
        X, Y = val_dataloader.dataset.get_all_data()
        
        if str(config['Threshold_optimizer']['epoch'])== 'best':
            print("Optimizing thresholds for best model...")
            conf.load_epoch = "best"

        elif str(config['Threshold_optimizer']['epoch'])== 'last': 
            print("Optimizing thresholds for last model...")
            conf.load_epoch = conf.n_epochs
        
        model = StarDist3D(conf)

    if (config['Threshold_optimizer']['NMS_thresh'] is not None and config['Threshold_optimizer']['IoU_thresh'] is not None):
        print('Using default settings for NMS threshold optimization...')
        model.optimize_thresholds(X, Y, 
                                    nms_threshs = config['Threshold_optimizer']['NMS_thresh'],
                                    iou_threshs = config['Threshold_optimizer']['IoU_thresh'])
    else:
        model.optimize_thresholds(X, Y)

if __name__ == '__main__':
    main()