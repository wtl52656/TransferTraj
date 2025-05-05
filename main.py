import os
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from models.transfertraj import TransferTraj, load_transfer_feature
from data import TransferTrajDataset, PretrainPadder, fetch_task_padder, X_COL, Y_COL, coord_transform_GPS_UTM
from pipeline import train_model, test_model, finetune_model
import warnings

warnings.filterwarnings('ignore')

SETTINGS_CACHE_DIR = os.environ.get('SETTINGS_CACHE_DIR', os.path.join('settings', 'cache'))
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
LOG_SAVE_DIR = os.environ.get('LOG_SAVE_DIR', 'logs')
PRED_SAVE_DIR = os.environ.get('PRED_SAVE_DIR', 'predictions')


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default = "local_test")
    parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=7)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    torch.multiprocessing.set_start_method('spawn')
    device = f'cuda' if torch.cuda.is_available() and args.cuda is not None else 'cpu'

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # print(device)
    # exit()
    # This key is an indicator of multiple things.
    datetime_key = utils.get_datetime_key()


    print(f'====START EXPERIMENT, DATETIME KEY: {datetime_key} ====')

    # Load the settings file, and save a backup in the cache directory.
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        settings = json.load(fp)
    utils.create_if_noexists(SETTINGS_CACHE_DIR)
    with open(os.path.join(SETTINGS_CACHE_DIR, f'{datetime_key}.json'), 'w') as fp:
        json.dump(settings, fp)

    
    # Iterate through the multiple settings.
    for setting_i, setting in enumerate(settings):
        print(f'===SETTING {setting_i}/{len(settings)}===')
        SAVE_NAME = setting.get('save_name', None)
        # Load and build the training and testing datasets.
        train_traj_df = pd.read_hdf(setting['dataset']['train_traj_df'], key='trips')
        test_traj_df = pd.read_hdf(setting['dataset']['test_traj_df'], key='trips')
        
        train_traj_df = utils.traj_clip(train_traj_df)
        test_traj_df = utils.traj_clip(test_traj_df)

        if "chengdu" in setting['dataset']['train_traj_df']:
            UTM_region = 48
        if "xian" in setting['dataset']['train_traj_df']:
            UTM_region = 49
             
        train_dataset = TransferTrajDataset(traj_df=train_traj_df, UTM_region=UTM_region)
        test_dataset = TransferTrajDataset(traj_df=test_traj_df, UTM_region=UTM_region)
        
        # Load the POIs' coordinates and textual embeddings.
        poi_df = pd.read_hdf(setting['dataset']['poi_df'], key='pois')
        poi_embed = torch.from_numpy(np.load(setting['dataset']['poi_embed'])).float().to(device)
        poi_coors = poi_df[[X_COL, Y_COL]].to_numpy()
        poi_coors = coord_transform_GPS_UTM(poi_coors, UTM_region)
        poi_coors = torch.tensor(poi_coors).float().to(device)

        road_df = pd.read_hdf(setting['dataset']['road_df'], key='road_info')
        road_embed = torch.from_numpy(np.load(setting['dataset']['road_embed'])).float().to(device)
        road_coors = road_df[["road_lng", "road_lat"]].to_numpy()
        road_coors = coord_transform_GPS_UTM(road_coors, UTM_region)
        road_coors = torch.tensor(road_coors).float().to(device)


        # Build the learnable model.
        transfertraj = TransferTraj(poi_embed=poi_embed, poi_coors=poi_coors, road_embed= road_embed, road_coors=road_coors, UTM_region=UTM_region,
                        **setting['transfertraj']).to(device)
        
        if 'pretrain' in setting:
            # Pretrain the model with data input and output controlled by the pretrain padder.
            if setting['pretrain'].get('load', False):
                # Load pretrained model from a previous instances with the same save name.
                transfertraj.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.pretrain'),
                                                  map_location=device))
            else:
                pretrain_dataloader = DataLoader(train_dataset,
                                                 collate_fn=PretrainPadder(
                                                    **setting['pretrain']['padder']),
                                                 **setting['pretrain']['dataloader'])
                pretrain_log, saved_model_state_dict = train_model(model=transfertraj, dataloader=pretrain_dataloader, device = device,
                                           **setting['pretrain']['config'])
                if setting['pretrain'].get('save', False):
                    # Save the pretrained model with the save name.
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    torch.save(saved_model_state_dict, os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.pretrain'))
                    utils.create_if_noexists(os.path.join(LOG_SAVE_DIR, SAVE_NAME))
                    pretrain_log.to_csv(os.path.join(LOG_SAVE_DIR, SAVE_NAME, 'pretrain.csv'))
        
        if 'finetune' in setting:
            downstreamtask = setting['finetune']['padder']['name']
            # Finetune the model with task-specific input and output.
            if setting['finetune'].get('load', False):
                transfertraj.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.pretrain'),
                                                  map_location=device))
            else:
                finetune_padder = fetch_task_padder(padder_name=setting['finetune']['padder']['name'],
                                                    padder_params=setting['finetune']['padder']['params'])
                finetune_dataloader = DataLoader(train_dataset, collate_fn=finetune_padder,
                                                 **setting['finetune']['dataloader'])
                finetune_log, saved_model_state_dict = finetune_model(model=transfertraj, dataloader=finetune_dataloader, device = device,
                                           **setting['finetune']['config'])
                if setting['finetune'].get('save', False):
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    torch.save(saved_model_state_dict, os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.{downstreamtask}'))
                    utils.create_if_noexists(os.path.join(LOG_SAVE_DIR, SAVE_NAME))
                    finetune_log.to_csv(os.path.join(LOG_SAVE_DIR, SAVE_NAME, f'{downstreamtask}.csv'))

        if 'test' in setting:
            downstreamtask = setting['test']['padder']['name']
            # Test the model with task-specific input and in a non-gradient environment.
            test_padder = fetch_task_padder(padder_name=setting['test']['padder']['name'],
                                            padder_params=setting['test']['padder']['params'])
            test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=test_padder,
                                         **setting['test']['dataloader'])
            predictions, targets = test_model(model=transfertraj, device = device, dataloader=test_dataloader, **setting['test']['config'])
            if setting['test'].get('save', False):
                # Save both task-specific input and output as numpy arrays.
                utils.create_if_noexists(os.path.join(PRED_SAVE_DIR, SAVE_NAME))
                np.savez(os.path.join(PRED_SAVE_DIR, SAVE_NAME, f'{downstreamtask}_predictions.npz'), *predictions)
                np.savez(os.path.join(PRED_SAVE_DIR, SAVE_NAME, f'{downstreamtask}_targets.npz'), *targets)


if __name__ == '__main__':
    main()
