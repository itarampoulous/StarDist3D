### Adding root directory to sys.path (list of directories python look in for packages and modules)
import sys, os
from pathlib import Path
import h5py
from skimage import io
import numpy as np
import time
import warnings
import argparse
import yaml

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from stardist_tools.csbdeep_utils import normalize
from src.data.utils import load_img

from src.models.config import Config3D
from src.models.stardist3d import StarDist3D
from glob import glob


def load_model_using_config(config:dict):
    """This functions takes the configuration dictionary and loads the pretrained model for inference.

    Parameters
    ----------
    config : dict
        configuration dictionary in the following format:
        
        Attributes:
            - checkpoints_dir: <checkpoints_dir>
            - name: <name_of_the_model>


        Dataset:
            - image_dir: <path/to/directory/with/imagesforinference/>
            - export_dir: <path/to/directory/to/export/the/masks/>


        Returns
        -------
            StarDist3D model loaded with pretrained weights to be used for inference.
    """

    conf = Config3D(
        name                           = config['Attributes']['name'],
        isTrain = False,

        # ========================= Inference ==================================

        use_gpu                        = True if torch.cuda.is_available()and config['Inference']['use_gpu'] else False,
        #gpu_ids                       = config['Inference']['gpu_ids'],
        use_amp                        = config['Inference']['use_amp'])

    #Load weights for last model from checkpoints_dir
    conf.checkpoints_dir = config['Attributes']['checkpoints_dir']
    conf.load_epoch = conf.n_epochs
    return StarDist3D(conf)

def infer_using_model(model, test_images_paths:list, export_dir:str):
    """This function takes a loaded pretrained model, performs inference on all images 
    based on the paths provided and exports the masks in the export_dir.
    
    Parameters
    ----------
    model : 
        StarDist3D model loaded with pretrained weights to be used for inference
    test_images_paths : list
        List of paths to the input images to be used for inference. 
    export_dir : str
        Path to an export directory. If the directory doesn't exist, it will be created.
    """

    # Create export directory if it doesn't exist already
    if not os.path.exists(export_dir):
        print('The directory does not exist. It will be created!')
        os.makedirs(export_dir)

    i = 0
    for image_path in tqdm(test_images_paths):
        
        # First normalize input image X 
        X = normalize( load_img(image_path).squeeze() )[np.newaxis]
        
        # Predict mask for X
        Y_pred = model.predict_instance(X)[0]
        
        # Save the predicted mask in the export directory
        io.imsave(export_dir + os.path.basename(test_images_paths[i].split('.tif')[0]) + '_prediction.tif', Y_pred, check_contrast= False)

        i = i + 1

def main():
    parser = argparse.ArgumentParser(description='Inference using StarDist3D model')
    parser.add_argument('--config', type=str, help='path to inference configuration file', required=True)

    # Load YAML file
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    model = load_model_using_config(config)

    
    test_images_paths = list(glob(config['Dataset']['image_dir'] + "/*.tif"))
    print(f"{len(test_images_paths)} images detected in directory")


    export_dir = config['Dataset']['export_dir'] + config['Attributes']['name'] +  '/last_model_epoch' + str(model.opt.epoch_count) + '/'
    infer_using_model(model = model, test_images_paths=test_images_paths, export_dir=export_dir)

    print(f"Predicted masks have been successfully exported in: {export_dir}")

if __name__ == '__main__':
    main()