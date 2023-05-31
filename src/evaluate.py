import os
import numpy as np

from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import re

from scipy.ndimage import center_of_mass

import torch


title = 'Results from Stadist3D_epochs400_v5_last'
anisotropy = np.array([4.54, 1, 1])

class centroids_config:
    def __init__(self, config):
        if config['Evaluation_on_centroids']['evaluate_on_centroids'] is None:
            self.evaluate = False
        else:
            self.evaluate = config['Evaluation_on_centroids']['evaluate_on_centroids']

        if self.evaluate:

            self.predictions_dir = config['Evaluation_on_centroids']['Parameters']['predictions_dir']
        
            if config['Evaluation_on_centroids']['Parameters']['pattern'] is not None:
                self.pattern = config['Evaluation_on_centroids']['Parameters']['pattern']
            else:
                self.pattern = '_t' #use default value
            
            self.anisotropy = config['Evaluation_on_centroids']['Parameters']['anisotropy']
            self.centroids_GT_path = config['Evaluation_on_centroids']['Parameters']['centroids_GT_path']

            if config['Evaluation_on_centroids']['output_dir'] is not None:
                self.output_dir = config['Evaluation_on_centroids']['output_dir']
            else:
                self.output_dir = self.predictions_dir + '/performance_metrics/'

class masks_config:
    def __init__(self, config):

        if config['Evaluation_on_masks']['evaluate_on_masks'] is None:
            self.evaluate = False
        else:
            self.evaluate = config['Evaluation_on_masks']['evaluate_on_masks']


        if self.evaluate:         

            self.predictions_dir = config['Evaluation_on_masks']['Parameters']['predictions_dir']

            if config['Evaluation_on_masks']['Parameters']['pattern_predictions_dir'] is not None:
                self.pattern_predictions = config['Evaluation_on_centroids']['Parameters']['pattern_predictions_dir']
            else:
                self.pattern_predictions = '_t' #use default value

                self.masks_GT_dir = config['Evaluation_on_masks']['Parameters']['masks_GT_dir']

                if config['Evaluation_on_masks']['Parameters']['pattern_masks_GT_dir'] is not None:
                    self.pattern_masks_GT = config['Evaluation_on_centroids']['Parameters']['pattern_masks_GT_dir']
                else:
                    self.pattern_masks_GT = '_t' #use default value        


                self.anisotropy = config['Evaluation_on_masks']['Parameters']['anisotropy']

                if config['Evaluation_on_masks']['output_dir'] is not None:
                    self.output_dir = config['Evaluation_on_masks']['output_dir']
                else:
                    self.output_dir = self.predictions_dir + '/performance_metrics/'

class evaluation_configuration:
    def __init__(self, config):
        self.centroids = centroids_config(config=config)
        self.masks = masks_config(config=config)


def extract_timepoint_from_filename(filename:str, pattern:str)->int:
    """This function uses a pattern to extract the timepoint from a given filename.

    Parameters
    ----------
    filename : str
        the name of a file
    pattern : str
        a string in the filename followed by the digits (timepoint) to be extracted

    Returns
    -------
    int
        the digits following the pattern in the filename (e.g., for the following filename 'embryo_t42' with pattern '_t', the function returns 42)

    Raises
    ------
    ValueError
        Raised when the pattern is not detected in the filename.
    """
    
    find_timepoint = re.search(pattern + '(\d+)', filename)

    if find_timepoint:
        return int(find_timepoint.group(1))
    
    else:
        raise ValueError(f'The pattern ({pattern}) was not found in the path basename.') 




def get_paths_and_timepoints(dir: str, pattern = '_t')->tuple:
    """This function detects all tif files in a given directory containing the model's predicted masks
    and returns a list of absolute paths of the TIF/TIFF files in this directory and extracts the timepoint
    from the filename using a provided pattern.

    Parameters
    ----------
    directory : str
        A directory containing the predicted masks in TIF/TIFF format to be used for evaluation of the model's performance.
        Note that the TIF files need to contain the timepoint after the pattern provided.
        For example, if the provided pattern is '_t', _t<timepoint> is expected on the filenames of the TIF/TIFF files and
        the digits following the pattern will be extracted, e.g., 42 will be extracted for 'embryo_t42.tif'

    pattern : str, optional
        _description_, by default '_t'

    Returns
    -------
    tuple
        predictions_paths: a list of the global paths to the predicted masks.
        predictions_timepoints: a list of timepoints extracted from predictions_paths for each mask.
    """
    paths = glob(dir + '/*.tif')

    timepoints = []
    for path in paths:
        timepoints.append(extract_timepoint_from_filename(os.path.basename(path), pattern = pattern))

    assert len(paths) == len(timepoints)

    # Sort based on timepoints
    paths, timepoints = zip(*sorted(zip(paths, timepoints), key=lambda x: x[1]))

    return paths, timepoints



def evaluate_using_centroidsGT(predictions_paths: list, timepoints: list, centroidsGT: pd.DataFrame, anisotropy: np.ndarray, output_dir:str)->pd.DataFrame:
    """This function evaluates the predictions using a list of ground truth centroids.
    It iterates through the predicted masks and assesses the number of centroids within each mask,
    distance between the ground truth centroid and the centroid of the predicted mask.
    If a cell name is known for a centroid unique to a predicted mask it associates it
    with the predicted mask.

    Parameters
    ----------
    predictions_paths : list
        a list of the global paths to the predicted masks.
    timepoints : list
        a list of timepoints for each path in predictions_paths.
    centroidsGT : pd.DataFrame
        a pandas DataFrame containing the timepoint ('time'), XYZ coordinates and cell names ('cell_name') (if available) for each centroid. 
    anisotropy: np.ndarray
        the anisotropy in the predicted masks as [z, y, x]
    output_dir : str
        the target directory to export the csv file containing the evaluation metrics, by default the csv file is exported in the directory containing the first mask.
    
    Returns
    -------
    pd.DataFrame
        csv file saved within the directory of predictions_paths containing:
        - 'time': timepoint
        - 'labelID': individual label ID (unique per timepoint)
        - 'Z', 'Y', 'X': coordinates of the centroid extracted from the predicted mask
        - 'area': area of the predicted mask
        - 'n_GTcentroids_within': number of centroids in the ground truth that fall within each mask
        - 'GT_Z', 'GT_Y', 'GT_X': coordinates of the centroid in ground truth
        - 'distance_fromGT': Euclidean distance of the predicted centroids (extracted from the predicted mask) from the ground truth centroid
        - 'cell_name': If a predicted mask only covers a single centroid and cell names (cell_name) are available in the centroidsGT DataFrame, the cell name of the ground truth centroid will be included.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device to evaluate model performance using centroids ground truth.')

    if output_dir is None:
        output_dir=os.path.dirname(predictions_paths[0])

    model_evaluation_centroidsGT = pd.DataFrame(columns=['time', 'labelID', 'Z', 'Y', 'X', 'area',
                                                  'n_GTcentroids_within', 'GT_Z', 'GT_Y', 'GT_X', 'distance_fromGT', 'cell_name'])

    x = 0
    for n_files in tqdm(range(len(predictions_paths)), desc='Evaluating predicted mask using ground truth centroids'):

        label = io.imread(predictions_paths[n_files])

        out_time = timepoints[n_files]

        if 'cell_name' in centroidsGT.columns:
            GT_cell_names = (centroidsGT[centroidsGT['time']==out_time].filter(['cell_name'])).values.tolist()


        GTcentr = (centroidsGT[centroidsGT['time']==out_time].filter(['Z', 'Y', 'X'])).values.tolist()
        GT_centr_torch = torch.tensor(GTcentr)
        GT_centr_torch.to(device)

        for labelID in tqdm(np.unique(label)):
            if labelID == 0:
                continue

            out_labelID = [int(labelID)]
            out_ZYX = [int(coord.round(0)) for coord in list(center_of_mass(label == labelID))]
            out_area = [np.sum(label == labelID)]
            
            label_torch = torch.tensor(np.argwhere(label == labelID).tolist())
            label_torch.to(device)

            GT_centroids_within = torch.stack([torch.any(torch.all(label_torch == tensor, dim=1)) for tensor in GT_centr_torch])

            out_n_GTcentroids_within = [(GT_centroids_within).sum().numpy()]


            if out_n_GTcentroids_within[0] == 1:
                out_GT_ZYX = GTcentr[np.where(GT_centroids_within)[0][0]]
                out_distance = [np.linalg.norm((np.array(out_ZYX) - np.array(out_GT_ZYX))*anisotropy)] #linear euclidean distance
                if 'cell_name' in centroidsGT.columns:
                    out_cell_name = [GT_cell_names[np.where(GT_centroids_within)[0][0]]]
                else:
                    out_cell_name = ['']

            else:
                out_GT_ZYX = ['','','']
                out_distance = ['']
                out_cell_name = ['']

            model_evaluation_centroidsGT.loc[x] = [out_time] + out_labelID + out_ZYX + out_area + out_n_GTcentroids_within + out_GT_ZYX + out_distance + out_cell_name
            model_evaluation_centroidsGT.to_csv(output_dir + '/evaluation_on_centroidsGT.csv', index = False)
            x = x + 1



def summary_metrics_on_centroidsGT(model_evaluation_centroidsGT:pd.DataFrame, centroidsGT:pd.DataFrame, output_dir:str)->pd.DataFrame:
    """Summarize the detailed model evaluation on centroids ground truth.

    Parameters
    ----------
    model_evaluation_centroidsGT : pd.DataFrame
        A pandas DataFrame containing the following information for each  mask predicted by the model.
        - 'time': timepoint
        - 'labelID': individual label ID (unique per timepoint)
        - 'Z', 'Y', 'X': coordinates of the centroid extracted from the predicted mask
        - 'area': area of the predicted mask
        - 'n_GTcentroids_within': number of centroids in the ground truth that fall within each mask
        - 'GT_Z', 'GT_Y', 'GT_X': coordinates of the centroid in ground truth
        - 'distance_fromGT': Euclidean distance of the predicted centroids (extracted from the predicted mask) from the ground truth centroid
        - 'cell_name': If a predicted mask only covers a single centroid and cell names (cell_name) are available in the centroidsGT DataFrame, the cell name of the ground truth centroid will be included.

    centroids_GT : pd.DataFrame
        a pandas DataFrame containing the timepoint ('time'), XYZ coordinates and cell names ('cell_name') (if available) for each centroid.
        Note that this has to match the centroids_GT used in the function evaluate_using_centroidsGT to generate the model_evaluation_centroidsGT.
    output_dir : str
        the target directory to export the csv file containing the evaluation metrics.

    Returns
    -------
    pd.DataFrame
        csv file saved within the <output_dir> containing the following information:
            - 'time': timepoint
            - 'n_cells_from_model': Total number of masks detected by the number per timepoint
            - 'TruePositives': Number of masks covering 1 centroid (TRUE POSITIVES) per timepoint
            - 'FalsePositives': Number of masks not covering any centroid (FALSE POSITIVES) per timepoint
            - 'More_than_1_centroid': Number of masks covering more than 1 centroids per timepoint
            - 'FalseNegatives': Centroids missed by the model (FALSE NEGATIVES) per timepoint
            - 'centroid_distance_mean': Mean Euclidean distance between the centroid extracted from the predicted mask by the model and the centroid in the ground truth per timepoint
            - 'centroid_distance_std': Standard deviation of the centroid_distance_mean
    """
    summary_metrics_centroidsGT = pd.DataFrame(columns=['time', 'n_cells_from_model', 'TruePositives', 'FalsePositives', 'More_than_1_centroid','FalseNegatives', 'centroid_distance_mean', 'centroid_distance_std'])
    summary_metrics_centroidsGT['time'] = model_evaluation_centroidsGT.groupby('time').count()['X'].index
    summary_metrics_centroidsGT.index = summary_metrics_centroidsGT['time']

    summary_metrics_centroidsGT['n_cells_from_model'] = model_evaluation_centroidsGT.groupby('time').count()['X']

    summary_metrics_centroidsGT['TruePositives'] = model_evaluation_centroidsGT[model_evaluation_centroidsGT['n_GTcentroids_within'] == 1].groupby('time').count()['n_GTcentroids_within'] # masks cover only 1 centroid (TRUE POSITIVE)
    summary_metrics_centroidsGT['TruePositives'] = summary_metrics_centroidsGT['TruePositives'].fillna(0)

    summary_metrics_centroidsGT['FalsePositives'] = model_evaluation_centroidsGT[model_evaluation_centroidsGT['n_GTcentroids_within'] == 0].groupby('time').count()['n_GTcentroids_within'] #masks that don't cover any centroid (FALSE POSITIVE)
    summary_metrics_centroidsGT['FalsePositives'] = summary_metrics_centroidsGT['FalsePositives'].fillna(0)

    summary_metrics_centroidsGT['More_than_1_centroid'] = model_evaluation_centroidsGT[model_evaluation_centroidsGT['n_GTcentroids_within'] > 1].groupby('time').count()['n_GTcentroids_within'] # masks cover more than 1 centroid
    
    summary_metrics_centroidsGT['More_than_1_centroid'] = summary_metrics_centroidsGT['More_than_1_centroid'].fillna(0)

    summary_metrics_centroidsGT['centroid_distance_mean'] = model_evaluation_centroidsGT[model_evaluation_centroidsGT['n_GTcentroids_within'] == 1].groupby('time').mean()['distance_fromGT'] #centroid Eucledian distance from tracking GT
    summary_metrics_centroidsGT['centroid_distance_std'] = model_evaluation_centroidsGT[model_evaluation_centroidsGT['n_GTcentroids_within'] == 1].groupby('time').std()['distance_fromGT'] #centroid Eucledian distance from tracking GT

    summary_metrics_centroidsGT = pd.merge(summary_metrics_centroidsGT, centroidsGT.groupby('time').count()['X'], left_on=summary_metrics_centroidsGT['time'], right_on=centroidsGT.groupby('time').count()['X'].index, how='outer').drop(['key_0'], axis=1).rename(columns={'X':'n_cells_from_GT'})
    summary_metrics_centroidsGT = summary_metrics_centroidsGT.reindex(columns=['time', 'n_cells_from_GT', 'n_cells_from_model', 'TruePositives', 'FalsePositives', 'More_than_1_centroid', 'FalseNegatives', 'centroid_distance_mean', 'centroid_distance_std'])
    
    summary_metrics_centroidsGT['FalseNegatives'] = np.maximum(summary_metrics_centroidsGT['n_cells_from_GT'] - summary_metrics_centroidsGT['TruePositives'] - summary_metrics_centroidsGT['More_than_1_centroid'], 0) #Difference of n_cells in GT vs n_cells in model. If negative, returns 0

    summary_metrics_centroidsGT.to_csv(output_dir + '/summary_metrics_on_centroidsGT.csv', index = False)

    return summary_metrics_centroidsGT

def plot_summary_metrics_on_centroidsGT(summary_metrics_centroidsGT:pd.DataFrame, output_dir:str, fs = 30):
    """Plot summary metrics generated by summary_metrics_on_centroidsGT. 

    Parameters
    ----------
    summary_metrics_centroidsGT : pd.DataFrame
        pandas DataFrame containing the following columns/information:
            - 'time': timepoint
            - 'n_cells_from_model': Total number of masks detected by the number per timepoint
            - 'TruePositives': Number of masks covering 1 centroid (TRUE POSITIVES) per timepoint
            - 'FalsePositives': Number of masks not covering any centroid (FALSE POSITIVES) per timepoint
            - 'More_than_1_centroid': Number of masks covering more than 1 centroids per timepoint
            - 'FalseNegatives': Centroids missed by the model (FALSE NEGATIVES) per timepoint
            - 'centroid_distance_mean': Mean Euclidean distance between the centroid extracted from the predicted mask by the model and the centroid in the ground truth per timepoint
            - 'centroid_distance_std': Standard deviation of the centroid_distance_mean
    output_dir : str
        the target directory to export the tif file containing the figures.
    fs : int, optional
        font size, by default 30
    """

    try:
        plt.style.use('seaborn-white')
    except:
        pass

    fig1, ax = plt.subplots(figsize = (15,10))

    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['n_cells_from_GT'], label = 'Number of centroids in ground truth', color = 'orange', lw = 4)
    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['n_cells_from_model'], label = 'Number of cells infered by the model', color = 'blue')


    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['TruePositives'], label = 'Masks covering 1 centroid (TRUE POSITIVES)', color = 'green', linestyle = '--')
    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['More_than_1_centroid'], label = 'Masks covering more than 1 centroids', color = 'magenta', linestyle = '--')
    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['FalsePositives'], label = 'Masks not covering any centroid (FALSE POSITIVES)', color = 'red', linestyle = '--')
    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['FalseNegatives'], label = 'Centroids missed by the model (FALSE NEGATIVES)', color = 'black', linestyle = '--')


    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Number of cells', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.tick_params(axis='both', labelsize=fs/1.5)
    ax.legend(fontsize = fs/1.5)
    fig1.savefig(output_dir + '/summary_metrics_on_centroidsGT.tif',dpi=300)

    fig2, ax = plt.subplots(figsize = (15,10))

    ax.plot(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['centroid_distance_mean'], label = 'distance of predicted mask centroids from ground truth', color = 'blue')
    ax.fill_between(summary_metrics_centroidsGT['time'], summary_metrics_centroidsGT['centroid_distance_mean'] - summary_metrics_centroidsGT['centroid_distance_std'], summary_metrics_centroidsGT['centroid_distance_mean'] + summary_metrics_centroidsGT['centroid_distance_std'], color='blue', alpha = 0.2, lw = 0)

    ax.set_ylim(bottom = 0)
    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Euclidean distance (in pixels)', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)
    ax.tick_params(axis='both', labelsize=fs/1.5)
    fig2.savefig(output_dir + '/summary_metrics_on_centroidsGT_Euclideandistance.tif',dpi=300)

def iou(mask1:np.ndarray, mask2:np.ndarray):
    """This function takes two binary masks and returns their intersection over union.

    Parameters
    ----------
    mask1 : np.ndarray
        a binary mask
    mask2 : np.ndarray
        another binary mask

    Returns
    -------
    float
        the intersection over union for the given masks
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1 + mask2) - intersection
    iou = intersection / union
    return iou

def extract_centroids_from_masks(masks_paths: list, masks_timepoints: list, output_dir:str=None):

    if output_dir is None:
        output_dir=os.path.dirname(masks_paths[0])

    centroids_from_masks = pd.DataFrame(columns=['time', 'labelID', 'Z', 'Y', 'X'])

    x = 0
    for n_files in tqdm(range(len(masks_paths))):

        label = io.imread(masks_paths[n_files])

        out_time = masks_timepoints[n_files]

        for labelID in tqdm(np.unique(label)):
            if labelID == 0:
                continue

            out_labelID = [int(labelID)]
            out_ZYX = [int(coord.round(0)) for coord in list(center_of_mass(label == labelID))]

            centroids_from_masks.loc[x] = [out_time] + out_labelID + out_ZYX
            # centroids_from_masks = centroids_from_masks.round()

            centroids_from_masks.to_csv(output_dir + '/centroids_from_masks.csv', index=False)
            x += 1

    return centroids_from_masks    


def evaluate_using_masksGT(predictions_paths: list, predictions_timepoints: list, masksGT_paths: list, masksGT_timepoints: list, anisotropy: np.ndarray, output_dir:str)->pd.DataFrame:
    """This function evaluates the predictions using a list of ground truth masks.
    It iterates through the predicted masks and assesses the number of centroids within each mask,
    distance between the centroids of the ground truth mask and the centroid of the predicted mask.
    It also calculates the intersection of union per timepoint. If a cell name is known for a centroid
    unique to a predicted mask it associates it with the predicted mask.

    Parameters
    ----------
    predictions_paths : list
        a list of the global paths to the predicted masks.
    predictions_timepoints : list
        a list of timepoints for each path in predictions_paths.
    masksGT_paths : list
        a list of the global paths to the ground truth masks.
    anisotropy: the anisotropy in the predicted masks as [z, y, x]

    output_dir : str
        the target directory to export the csv file containing the evaluation metrics, by default the csv file is exported in the directory containing the first predicted mask.
    
    Returns
    -------
    pd.DataFrame
        csv file saved within the directory of predictions_paths containing:
        - 'time': timepoint
        - 'labelID': individual label ID (unique per timepoint)
        - 'Z', 'Y', 'X': coordinates of the centroid extracted from the predicted mask
        - 'area': area of the predicted mask
        - 'n_GTcentroids_within': number of centroids extracted from the ground truth masks that fall within each mask
        - 'GT_labelID': individual label ID for the ground truth mask
        - 'GT_Z', 'GT_Y', 'GT_X': coordinates of the centroid extracted from the ground truth masks that fall within a given mask
        - 'distance_fromGT': Euclidean distance of the predicted centroids (extracted from the predicted mask) from the ground truth centroid extracted from the ground truth masks
        - 'IoU': intersection over union for the two masks
        - 'cell_name': If a predicted mask only covers a single centroid and cell names (cell_name) are available in the centroidsGT DataFrame, the cell name of the ground truth centroid will be included.
    """

    # Sort based on timepoints
    predictions_paths, predictions_timepoints = zip(*sorted(zip(predictions_paths, predictions_timepoints), key=lambda x: x[1]))
    masksGT_paths, masksGT_timepoints = zip(*sorted(zip(masksGT_paths, masksGT_timepoints), key=lambda x: x[1]))
    assert predictions_timepoints == masksGT_timepoints, 'The timepoints for the predicted masks do not match the timepoints for the ground truth masks given.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device to evaluate model performance using masks ground truth.')

    if output_dir is None:
        output_dir=os.path.dirname(predictions_paths[0])

    model_evaluation_masksGT = pd.DataFrame(columns=['time', 'labelID', 'Z', 'Y', 'X', 'area',
                                                  'n_GTcentroids_within', 'GT_labelID', 'GT_Z', 'GT_Y', 'GT_X', 'distance_fromGT', 'IoU', 'cell_name'])

    GT_mask_centroids_all = pd.DataFrame(columns=['time', 'GT_labelID', 'GT_Z', 'GT_Y', 'GT_X'])

    #############

    x = 0
    
    for n_files in tqdm(range(len(predictions_paths)), desc = 'Evaluating predicted mask using ground truth masks'):

        out_time = predictions_timepoints[n_files]

        label = io.imread(predictions_paths[n_files])
        GT_label = io.imread(masksGT_paths[n_files])

        # Get centroids for the ground truth mask from a single timepoint (out_time)
        GT_mask_centroids = pd.DataFrame(columns=['time', 'GT_labelID', 'GT_Z', 'GT_Y', 'GT_X'])
        j = 0
        for GT_labelID in np.unique(GT_label):
            if GT_labelID == 0:
                continue

            out_GT_labelID = [int(GT_labelID)]
            
            out_GT_ZYX = [int(coord.round(0)) for coord in list(center_of_mass(GT_label == GT_labelID))]

            GT_mask_centroids.loc[j] = [out_time] + out_GT_labelID + out_GT_ZYX

            j += 1

        GT_mask_centroids_all = pd.concat([GT_mask_centroids_all, GT_mask_centroids], axis = 0)
        GT_mask_centroids_all.to_csv(output_dir + '/Masks_GTcentroids.csv', index = False)

        GTcentr = (GT_mask_centroids.filter(['GT_Z', 'GT_Y', 'GT_X'])).values.tolist()
        GT_labelID = (GT_mask_centroids.filter(['GT_labelID'])).values.tolist()

        if 'cell_name' in GT_mask_centroids.columns:
            GT_cell_names = (GT_mask_centroids[GT_mask_centroids['time']==out_time].filter(['cell_name'])).values.tolist()


        # GTcentr = (centroidsGT[centroidsGT['time']==out_time].filter(['Z', 'Y', 'X'])).values.tolist()
        GT_centr_torch = torch.tensor(GTcentr)
        GT_centr_torch.to(device)

        for labelID in tqdm(np.unique(label)):
            if labelID == 0:
                continue

            out_labelID = [int(labelID)]
            out_ZYX = [int(coord.round(0)) for coord in list(center_of_mass(label == labelID))]            
            out_area = [np.sum(label == labelID)]
            
            label_torch = torch.tensor(np.argwhere(label == labelID).tolist())
            label_torch.to(device)

            GT_centroids_within = torch.stack([torch.any(torch.all(label_torch == tensor, dim=1)) for tensor in GT_centr_torch])
            out_n_GTcentroids_within = [(GT_centroids_within).sum().numpy()]


            if out_n_GTcentroids_within[0] == 1:
                out_GT_labelID = GT_labelID[np.where(GT_centroids_within)[0][0]]
                out_GT_ZYX = GTcentr[np.where(GT_centroids_within)[0][0]]
                out_distance = [np.linalg.norm((np.array(out_ZYX) - np.array(out_GT_ZYX))*anisotropy)] #linear euclidean distance
                
                out_IoU = [iou(np.int0(label == labelID), np.int0(GT_label == out_GT_labelID))]

                
                if 'cell_name' in GT_mask_centroids.columns:
                    out_cell_name = [GT_cell_names[np.where(GT_centroids_within)[0][0]]]
                else:
                    out_cell_name = ['']

            else:
                out_GT_labelID = ['']
                out_GT_ZYX = ['','','']
                out_distance = ['']
                out_cell_name = ['']
                out_IoU = ['']

            model_evaluation_masksGT.loc[x] = [out_time] + out_labelID + out_ZYX + out_area + out_n_GTcentroids_within + out_GT_labelID + out_GT_ZYX + out_distance + out_IoU + out_cell_name
            model_evaluation_masksGT.to_csv(output_dir + '/evaluation_on_masksGT.csv', index = False)
            x = x + 1


def summary_metrics_on_masksGT(model_evaluation_masksGT:pd.DataFrame, mask_centroidsGT:pd.DataFrame, output_dir:str)->pd.DataFrame:
    """Summarize the detailed model evaluation on centroids ground truth.

    Parameters
    ----------
    model_evaluation_masksGT : pd.DataFrame
        A pandas DataFrame containing the following information for each  mask predicted by the model.
        - 'time': timepoint
        - 'labelID': individual label ID (unique per timepoint)
        - 'Z', 'Y', 'X': coordinates of the centroid extracted from the predicted mask
        - 'area': area of the predicted mask
        - 'n_GTcentroids_within': number of centroids in the ground truth that fall within each mask
        - 'GT_labelID': 
        - 'GT_Z', 'GT_Y', 'GT_X': coordinates of the centroid in ground truth
        - 'distance_fromGT': Euclidean distance of the predicted centroids (extracted from the predicted mask) from the ground truth centroid
        - 'IoU': intersection over union
        - 'cell_name': If a predicted mask only covers a single centroid and cell names (cell_name) are available in the centroidsGT DataFrame, the cell name of the ground truth centroid will be included.

    mask_centroidsGT: pd.DataFrame
        a pandas DataFrame containing the timepoint ('time'), XYZ coordinates and cell names ('cell_name') (if available) for each centroid.
        Note that this has to match the mask_centroidsGT generated by the model_evaluation_masksGT.
    output_dir : str
        the target directory to export the csv file containing the evaluation metrics.

    Returns
    -------
    pd.DataFrame
        csv file saved within the <output_dir> containing the following information:
            - 'time': timepoint
            - 'n_cells_from_model': Total number of masks detected by the number per timepoint
            - 'TruePositives': Number of masks covering 1 centroid (TRUE POSITIVES) per timepoint
            - 'FalsePositives': Number of masks not covering any centroid (FALSE POSITIVES) per timepoint
            - 'More_than_1_centroid': Number of masks covering more than 1 centroids per timepoint
            - 'FalseNegatives': Centroids missed by the model (FALSE NEGATIVES) per timepoint
            - 'centroid_distance_mean': Mean Euclidean distance between the centroid extracted from the predicted mask by the model and the centroid in the ground truth per timepoint
            - 'centroid_distance_std': Standard deviation of the centroid_distance_mean
            - 'IoU_mean': Mean intersection over union between the predicted mask by the model and the ground truth mask
            - 'IoU_std': standard deviation of the IoU_mean
    """
    summary_metrics_masksGT = pd.DataFrame(columns=['time', 'n_cells_from_model', 'TruePositives', 'FalsePositives', 'More_than_1_centroid','FalseNegatives', 'centroid_distance_mean', 'centroid_distance_std', 'IoU_mean', 'IoU_std'])

    summary_metrics_masksGT['time'] = model_evaluation_masksGT.groupby('time').count()['X'].index
    summary_metrics_masksGT.index = summary_metrics_masksGT['time']

    summary_metrics_masksGT['n_cells_from_model'] = model_evaluation_masksGT.groupby('time').count()['X']

    summary_metrics_masksGT['TruePositives'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 1].groupby('time').count()['n_GTcentroids_within'] # masks cover only 1 centroid (TRUE POSITIVE)
    summary_metrics_masksGT['TruePositives'] = summary_metrics_masksGT['TruePositives'].fillna(0)

    summary_metrics_masksGT['FalsePositives'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 0].groupby('time').count()['n_GTcentroids_within'] #masks that don't cover any centroid (FALSE POSITIVE)
    summary_metrics_masksGT['FalsePositives'] = summary_metrics_masksGT['FalsePositives'].fillna(0)

    summary_metrics_masksGT['More_than_1_centroid'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] > 1].groupby('time').count()['n_GTcentroids_within'] # masks cover more than 1 centroid
    summary_metrics_masksGT['More_than_1_centroid'] = summary_metrics_masksGT['More_than_1_centroid'].fillna(0)

    summary_metrics_masksGT['centroid_distance_mean'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 1].filter(['time', 'distance_fromGT']).groupby('time').mean()['distance_fromGT'] #centroid Eucledian distance from masks GT centroids
    summary_metrics_masksGT['centroid_distance_std'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 1].filter(['time', 'distance_fromGT']).groupby('time').std()['distance_fromGT']

    summary_metrics_masksGT['IoU_mean'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 1].filter(['time', 'IoU']).groupby('time').mean()['IoU'] #mean intersection over union between the ground truth and the predicted masks
    summary_metrics_masksGT['IoU_std'] = model_evaluation_masksGT[model_evaluation_masksGT['n_GTcentroids_within'] == 1].filter(['time', 'IoU']).groupby('time').std()['IoU'] #std intersection over union between the ground truth and the predicted masks


    summary_metrics_masksGT = pd.merge(summary_metrics_masksGT, mask_centroidsGT.groupby('time').count()['GT_X'], left_on=summary_metrics_masksGT['time'], right_on=mask_centroidsGT.groupby('time').count()['GT_X'].index, how='inner').drop(['key_0'], axis=1).rename(columns={'GT_X':'n_cells_from_GT'})
    summary_metrics_masksGT = summary_metrics_masksGT.reindex(columns=['time', 'n_cells_from_GT', 'n_cells_from_model', 'TruePositives', 'FalsePositives', 'More_than_1_centroid', 'FalseNegatives', 'centroid_distance_mean', 'centroid_distance_std', 'IoU_mean', 'IoU_std'])
    
    summary_metrics_masksGT['FalseNegatives'] = np.maximum(summary_metrics_masksGT['n_cells_from_GT'] - summary_metrics_masksGT['TruePositives'] - summary_metrics_masksGT['More_than_1_centroid'], 0) #Difference of n_cells in GT vs n_cells in model. If negative, returns 0

    summary_metrics_masksGT.to_csv(output_dir + '/summary_metrics_masksGT.csv', index = False)

    return summary_metrics_masksGT


def plot_summary_metrics_on_masksGT(summary_metrics_masksGT:pd.DataFrame, output_dir:str, fs = 30):
    """Plot summary metrics generated by summary_metrics_on_masksGT. 

    Parameters
    ----------
    summary_metrics_centroidsGT : pd.DataFrame
        pandas DataFrame containing the following columns/information:
            - 'time': timepoint
            - 'n_cells_from_model': Total number of masks detected by the number per timepoint
            - 'TruePositives': Number of masks covering 1 centroid (TRUE POSITIVES) per timepoint
            - 'FalsePositives': Number of masks not covering any centroid (FALSE POSITIVES) per timepoint
            - 'More_than_1_centroid': Number of masks covering more than 1 centroids per timepoint
            - 'FalseNegatives': Centroids missed by the model (FALSE NEGATIVES) per timepoint
            - 'centroid_distance_mean': Mean Euclidean distance between the centroid extracted from the predicted mask by the model and the centroid in the ground truth per timepoint
            - 'centroid_distance_std': Standard deviation of the centroid_distance_mean
    output_dir : str
        the target directory to export the tif file containing the figures.
    fs : int, optional
        font size, by default 30
    """

    try:
        plt.style.use('seaborn-white')
    except:
        pass

    fig1, ax = plt.subplots(figsize = (15,10))

    width = fig1.get_size_inches()[0]/(2.5*len(summary_metrics_masksGT['time']))

    ax.bar(x = summary_metrics_masksGT['time'].index - (width/2), height= summary_metrics_masksGT['n_cells_from_GT'], label = 'Number of centroids in ground truth', color = 'orange', width = width)
    ax.bar(x = summary_metrics_masksGT['time'].index + (width/2), height= summary_metrics_masksGT['n_cells_from_model'], label = 'Number of cells infered by the model', color = 'blue', width = width)
    ax.set_xticks(summary_metrics_masksGT['time'].index)
    ax.set_xticklabels(summary_metrics_masksGT['time'])

    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Number of cells', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)
    ax.tick_params(axis='both', labelsize=fs/1.5)
    fig1.savefig(output_dir + '/summary_metrics_masksGT_totalcount.tif',dpi=300)

    fig2, ax = plt.subplots(figsize = (15,10))
    
    width = fig2.get_size_inches()[0]/(2.5*len(summary_metrics_masksGT['time']))

    ax.bar(x = summary_metrics_masksGT['time'].index - (width/2), height= summary_metrics_masksGT['n_cells_from_GT'], label = 'Number of centroids in ground truth', color = 'orange', width = width)


    ax.bar(summary_metrics_masksGT['time'].index + (width/2),
           summary_metrics_masksGT['TruePositives'], 
           label = 'Masks covering 1 centroid (TRUE POSITIVES)', color = 'green', width = width)
    
    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['More_than_1_centroid'], bottom = summary_metrics_masksGT['TruePositives'], 
            label = 'Masks covering more than 1 centroids', color = 'magenta', width = width)    

    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['FalseNegatives'], bottom = summary_metrics_masksGT['TruePositives'] + summary_metrics_masksGT['More_than_1_centroid'], 
           label = 'Centroids missed by the model (FALSE NEGATIVES)', color = 'black', width = width) 

    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['FalsePositives'], bottom = summary_metrics_masksGT['TruePositives'] + summary_metrics_masksGT['FalseNegatives'] + summary_metrics_masksGT['More_than_1_centroid'], 
           label = 'Masks not covering any centroid (FALSE POSITIVES)', color = 'red', width = width)
    

    ax.set_xticks(summary_metrics_masksGT['time'].index)
    ax.set_xticklabels(summary_metrics_masksGT['time'])

    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Number of cells', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)
    ax.tick_params(axis='both', labelsize=fs/1.5)

    fig2.savefig(output_dir + '/summary_metrics_masksGT.tif',dpi=300)

    fig3, ax = plt.subplots(figsize = (15,10))
    
    width = fig3.get_size_inches()[0]/(2.5*len(summary_metrics_masksGT['time']))

    total = summary_metrics_masksGT['n_cells_from_GT']

    ax.bar(x = summary_metrics_masksGT['time'].index - (width/2), height= summary_metrics_masksGT['n_cells_from_GT']/total, label = 'Number of centroids in ground truth', color = 'orange', width = width)


    ax.bar(summary_metrics_masksGT['time'].index + (width/2),
           summary_metrics_masksGT['TruePositives']/total, 
           label = 'Masks covering 1 centroid (TRUE POSITIVES)', color = 'green', width = width)
    
    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['More_than_1_centroid']/total, bottom = summary_metrics_masksGT['TruePositives']/total, 
            label = 'Masks covering more than 1 centroids', color = 'magenta', width = width)    


    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['FalseNegatives']/total, bottom = summary_metrics_masksGT['TruePositives']/total + summary_metrics_masksGT['More_than_1_centroid']/total, 
           label = 'Centroids missed by the model (FALSE NEGATIVES)', color = 'black', width = width) 

    ax.bar(summary_metrics_masksGT['time'].index + (width/2), 
           summary_metrics_masksGT['FalsePositives']/total, bottom = summary_metrics_masksGT['TruePositives']/total + summary_metrics_masksGT['FalseNegatives']/total + summary_metrics_masksGT['More_than_1_centroid']/total, 
           label = 'Masks not covering any centroid (FALSE POSITIVES)', color = 'red', width = width)
    


    ax.set_xticks(summary_metrics_masksGT['time'].index)
    ax.set_xticklabels(summary_metrics_masksGT['time'])

    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Frequency', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)
    ax.tick_params(axis='both', labelsize=fs/1.5)

    fig3.savefig(output_dir + '/summary_metrics_masksGT_frequencies.tif',dpi=300)


    fig4, ax = plt.subplots(figsize = (15,10))

    ax.bar(summary_metrics_masksGT['time'].index, summary_metrics_masksGT['centroid_distance_mean'], label = 'distance of predicted mask centroids from ground truth', color = 'blue')
    ax.errorbar(summary_metrics_masksGT['time'].index, summary_metrics_masksGT['centroid_distance_mean'], yerr=summary_metrics_masksGT['centroid_distance_std'], fmt='o', color='black')
    ax.set_xticks(summary_metrics_masksGT['time'].index)
    ax.set_xticklabels(summary_metrics_masksGT['time'], fontsize = fs/1.5)
    ax.tick_params(axis='y', labelsize=fs/1.5)


    ax.set_ylim(bottom = 0)
    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Euclidean distance (in pixels)', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)
    fig4.savefig(output_dir + '/summary_metrics_masksGT_Euclideandistance.tif',dpi=300)


    fig5, ax = plt.subplots(figsize = (15,10))

    ax.bar(summary_metrics_masksGT['time'].index, summary_metrics_masksGT['IoU_mean'], label = 'Intersection over union between the predicted masks and the ground truth masks', color = 'blue')
    ax.errorbar(summary_metrics_masksGT['time'].index, summary_metrics_masksGT['IoU_mean'], yerr=summary_metrics_masksGT['IoU_std'], fmt='o', color='black')

    ax.set_xticks(summary_metrics_masksGT['time'].index)
    ax.set_xticklabels(summary_metrics_masksGT['time'], fontsize = fs/1.5)
    ax.tick_params(axis='y', labelsize=fs/1.5)

    ax.set_ylim([0,1])
    ax.set_xlabel('Timepoint', fontsize = fs)
    ax.set_ylabel('Intersection over union (IoU)', fontsize = fs)
    ax.set_title('Results', fontsize = fs)
    ax.legend(fontsize = fs/1.5)

    Mean_IoU = np.mean(summary_metrics_masksGT['IoU_mean'])
    ax.axhline(Mean_IoU, linestyle = 'dashed', color="black", lw=1.5, alpha = 0.5, label='Mean IoU')
    ax.text(x=fig5.get_size_inches()[0], y=Mean_IoU + 0.01, s=f'Mean IoU = {Mean_IoU.round(2)}', 
            size=fs/1.5, ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', edgecolor = 'white', boxstyle='round,pad=0.2', alpha = 0.7))
    ax.axhline(0.5, linestyle = 'dashed', color="lightgrey", lw=1.5, alpha = 0.5, label='Usual cut-off')

    fig5.savefig(output_dir + '/summary_metrics_masksGT_IoU.tif',dpi=300)

def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Evaluation of StarDist3D model performance')
    parser.add_argument('--config', type=str, help='path to evaluation configuration file', required=True)

    # Load YAML file
    with open(parser.parse_args().config, 'r') as f:
        yml_configuration = yaml.safe_load(f)
    
    print(yml_configuration)

    config = evaluation_configuration(yml_configuration)

    if config.centroids.evaluate:

        centroidsGT = pd.read_csv(config.centroids.centroids_GT_path,sep='\t')

        predictions_paths, timepoints = get_paths_and_timepoints(dir = config.centroids.predictions_dir, pattern = config.centroids.pattern)

        # Create export directory if it doesn't exist already
        if not os.path.exists(config.centroids.output_dir):
            os.makedirs(config.centroids.output_dir)
        

        model_evaluation_centroidsGT = evaluate_using_centroidsGT(predictions_paths=predictions_paths,
                                                                  timepoints=timepoints,
                                                                  centroidsGT=centroidsGT,
                                                                  anisotropy=config.centroids.anisotropy,
                                                                  output_dir=config.centroids.output_dir)

        model_evaluation_centroidsGT = pd.read_csv(config.centroids.output_dir + '/evaluation_on_centroidsGT.csv')


        summary_metrics_centroidsGT = summary_metrics_on_centroidsGT(model_evaluation_centroidsGT=model_evaluation_centroidsGT,
                                                                        centroidsGT=centroidsGT,output_dir=config.centroids.output_dir)

        plot_summary_metrics_on_centroidsGT(summary_metrics_centroidsGT = summary_metrics_centroidsGT, output_dir= config.centroids.output_dir)

    if config.masks.evaluate:
        print(config.masks.predictions_dir)
        
        masksGT_paths, masksGT_timepoints = get_paths_and_timepoints(dir = config.masks.masks_GT_dir)

        predictions_paths, predictions_timepoints = get_paths_and_timepoints(dir = config.masks.predictions_dir)

        # Create export directory if it doesn't exist already
        if not os.path.exists(config.masks.output_dir):
            os.makedirs(config.masks.output_dir)

        model_evaluation_masksGT =  evaluate_using_masksGT(predictions_paths=predictions_paths,
                               predictions_timepoints=predictions_timepoints,
                               masksGT_paths=masksGT_paths,
                               masksGT_timepoints=masksGT_timepoints,
                               anisotropy = config.masks.anisotropy,
                               output_dir=config.masks.output_dir)

        model_evaluation_masksGT = pd.read_csv(config.masks.output_dir + '/evaluation_on_masksGT.csv')

        mask_centroidsGT = pd.read_csv(config.masks.output_dir + '/Masks_GTcentroids.csv')

        summary_metrics_masksGT = summary_metrics_on_masksGT(model_evaluation_masksGT=model_evaluation_masksGT,
                                                                mask_centroidsGT=mask_centroidsGT,
                                                                output_dir=config.masks.output_dir)
        
        plot_summary_metrics_on_masksGT(summary_metrics_masksGT=summary_metrics_masksGT,
                                        output_dir=config.masks.output_dir)

if __name__=="__main__":
    main()

