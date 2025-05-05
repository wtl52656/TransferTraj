import math
import random

from einops import repeat
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd



TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
X_norm_COL = "first_lng"
Y_norm_COL = "first_lat"
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
FEATURE_PAD = 0
ST_MAP = {
    "spatial": [0, 1],
    "temporal": [2, 3]
}

KNOWN_TOKEN = 0
MASK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
UNKNOWN_TOKEN = 4
PAD_TOKEN = 5


def coord_transform_GPS_UTM(traj, UTM_region, origin_coord = "latlong", dest_coord = "utm"):
    from pyproj import Proj, transform
    
    if origin_coord == "latlong":
        origin = Proj(proj="latlong", datum="WGS84")
        dest = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区
        
    elif origin_coord == "utm":
        dest = Proj(proj="latlong", datum="WGS84")
        origin = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区

    else:
        raise NotImplementedError(f'coord type error')

    if traj.ndim == 2:
        easting, northing = transform(origin, dest, traj[:,0], traj[:,1])
        traj[:,0] = easting
        traj[:,1] = northing
    elif traj.ndim == 3:
        easting, northing = transform(origin, dest, traj[:,:,0], traj[:,:,1])
        traj[:,:,0] = easting
        traj[:,:,1] = northing
    return traj

class TransferTrajDataset(Dataset):
    def __init__(self, traj_df, UTM_region):
        """
        Dataset supporter for the Trajectory Learning Model.

        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()

        self.traj_df = traj_df
        
        self.UTM_region = UTM_region
        self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

        # 进行缩放操作
        traj_gps = traj_df[[X_COL, Y_COL]].values
        traj_utm = coord_transform_GPS_UTM(traj_gps, self.UTM_region)  # 转换到了以m为坐标系的单位
        self.traj_df[[X_COL, Y_COL]] = pd.DataFrame(traj_utm)
        self.traj_df[[X_norm_COL, Y_norm_COL]] = pd.DataFrame(traj_utm)

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        one_traj[X_COL] = one_traj[X_COL] - one_traj[X_COL].iloc[0]
        one_traj[Y_COL] = one_traj[Y_COL] - one_traj[Y_COL].iloc[0]
        return one_traj


class PretrainPadder:
    """Collate function for the pre-training.
    """

    def __init__(self, span_div_ratio, span_mask_ratio, feature_mask_prob, teacher_ratio=0.5):
        self.span_div_ratio = span_div_ratio
        self.span_mask_ratio = span_mask_ratio
        self.feature_mask_prob = feature_mask_prob
        self.teacher_ratio = teacher_ratio

    def __call__(self, raw_batch):
        """
        A function for padding the provided raw batch into a standard array.

        Features:
            0 - Longitude
            1 - Latitude
            2 - Timestamp
            3 - Timestamp delta
        Every feature is 2D: the first dimension is the value, the second is the token.

        Args:
            raw_batch (list): each item is a pd.DataFrame representing one trajectory.

        Returns:
            np.array: the padded input array, with shape (B, L, F, 2)
            np.array: the padded output array, with shape (B, L, F, 2)
            np.array: the padded dual-layer positions, with shape (B, L, 2)
        """
        input_batch, output_batch, pos_batch = [], [], []
        first_point = []
        for traj in raw_batch:
            prompt_arr, src_arr, trg_arr, prompt_pos, gen_pos = [], [], [], [], []
            prompt = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()
            _UTM = traj[[X_norm_COL, Y_norm_COL]].to_numpy()
            first_point.append(_UTM[0])
            valid_len = prompt.shape[0]

            span_div_idx = sorted(set(np.random.choice(valid_len, math.ceil(valid_len * self.span_div_ratio), replace=False)) |
                                  set([0, valid_len]))
            spans_idx = np.stack([span_div_idx[:-1], span_div_idx[1:]], axis=1)
            masked_span_i = np.random.choice(spans_idx.shape[0], math.ceil(
                spans_idx.shape[0] * self.span_mask_ratio), replace=False)


            input_row = np.stack([prompt, np.ones_like(prompt) * KNOWN_TOKEN], -1)  # (L, F, 2)
            output_row = np.copy(input_row)  # 对应的标签，也是 L，F，2

            # 设置轨迹点/轨迹片段mask
            mask_index = []
            for i, (l, r) in enumerate(spans_idx):
                if i in masked_span_i:  #这一段轨迹点的所有特征均被mask
                    mask_index = mask_index + list(range(l, r))  # 这些轨迹点的所有时间信息、空间信息均被mask
            input_row[np.ix_(mask_index, ST_MAP['spatial'], [0])] = FEATURE_PAD
            input_row[np.ix_(mask_index, ST_MAP['spatial'], [1])] = MASK_TOKEN
            input_row[np.ix_(mask_index, ST_MAP['temporal'], [0])] = FEATURE_PAD
            input_row[np.ix_(mask_index, ST_MAP['temporal'], [1])] = MASK_TOKEN
            output_row[np.ix_(mask_index, ST_MAP['spatial'], [1])] = UNKNOWN_TOKEN
            output_row[np.ix_(mask_index, ST_MAP['temporal'], [1])] = UNKNOWN_TOKEN


            # 设置部分轨迹特征mask
            prompt_mask = np.random.rand(input_row.shape[0]) < self.feature_mask_prob
            spatial_mask = np.random.rand(input_row.shape[0]) < 0.5
            temporal_mask = ~spatial_mask
            spatial_mask = repeat(prompt_mask & spatial_mask, 'L -> L F', F=input_row.shape[1])
            spatial_mask[:, ST_MAP['temporal']] = False
            temporal_mask = repeat(prompt_mask & temporal_mask, 'L -> L F', F=input_row.shape[1])
            temporal_mask[:, ST_MAP['spatial']] = False
            input_row[spatial_mask, 0] = FEATURE_PAD
            input_row[spatial_mask, 1] = MASK_TOKEN
            input_row[temporal_mask, 0] = FEATURE_PAD
            input_row[temporal_mask, 1] = MASK_TOKEN
            output_row[spatial_mask, 1] = UNKNOWN_TOKEN
            output_row[temporal_mask, 1] = UNKNOWN_TOKEN
            
            # 设置位置
            pos_row = np.arange(input_row.shape[0])
            # print(pos_row)
            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)
            # print(teacher, input_batch[0].shape, output_batch[0].shape, pos_batch[0].shape)
            
            # exit()
        # Pad the input and output arrays
        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()

        first_point = torch.tensor(first_point).float()
        
        return input_batch, output_batch, pos_batch, first_point

class TRecPadder:
    """Collate function for the Trajectory Prediction (TP) task.
    """

    def __init__(self, keep_ratio=0.125, eval=True):
        self.keep_ratio = keep_ratio
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        first_point = []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            _UTM = traj[[X_norm_COL, Y_norm_COL]].to_numpy()
            first_point.append(_UTM[0])


            input_row = np.stack([traj_feats, np.ones_like(traj_feats) * KNOWN_TOKEN], -1)  # (L, F, 2)
            output_row = np.copy(input_row)
            # pos_row = np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1)
            pos_row = np.arange(input_row.shape[0])
            # input_row[-self.pred_len:, ST_MAP['spatial'], 0] = FEATURE_PAD
            # input_row[-self.pred_len:, ST_MAP['spatial'], 1] = MASK_TOKEN
            # output_row[-self.pred_len:, ST_MAP['spatial'], 1] = UNKNOWN_TOKEN

            traj_i_mask = list(range(input_row.shape[0]))
            if (input_row.shape[0] - 1) % int(1 / self.keep_ratio) == 0:
                src_traj_index = traj_i_mask[::int(1 / self.keep_ratio)]
            else:
                src_traj_index = traj_i_mask[::int(1 / self.keep_ratio)] + [traj_i_mask[-1]]
            mask_index = list(set(traj_i_mask) - set(src_traj_index))
            
            input_row[np.ix_(mask_index, ST_MAP['spatial'], [0])] = FEATURE_PAD
            input_row[np.ix_(mask_index, ST_MAP['spatial'], [1])] = MASK_TOKEN
            output_row[np.ix_(mask_index, ST_MAP['spatial'], [1])] = UNKNOWN_TOKEN
            
            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)  # (L, 2)

        input_tensor = torch.tensor(pad_batch_3d(input_batch)).float()
        output_tensor = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_tensor = torch.tensor(pad_batch_2d(pos_batch)).long()
        first_point = torch.tensor(first_point).float()

        return input_tensor, output_tensor, pos_tensor, first_point


class TpPadder:
    """Collate function for the Trajectory Prediction (TP) task.
    """

    def __init__(self, pred_len, eval=True):
        self.pred_len = pred_len
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        first_point = []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            _UTM = traj[[X_norm_COL, Y_norm_COL]].to_numpy()
            first_point.append(_UTM[0])


            input_row = np.stack([traj_feats, np.ones_like(traj_feats) * KNOWN_TOKEN], -1)  # (L, F, 2)
            output_row = np.copy(input_row)
            # pos_row = np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1)
            pos_row = np.arange(input_row.shape[0])
            input_row[-self.pred_len:, ST_MAP['spatial'], 0] = FEATURE_PAD
            input_row[-self.pred_len:, ST_MAP['spatial'], 1] = MASK_TOKEN
            output_row[-self.pred_len:, ST_MAP['spatial'], 1] = UNKNOWN_TOKEN
            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)  # (L, 2)

        input_tensor = torch.tensor(pad_batch_3d(input_batch)).float()
        output_tensor = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_tensor = torch.tensor(pad_batch_2d(pos_batch)).long()
        first_point = torch.tensor(first_point).float()

        return input_tensor, output_tensor, pos_tensor, first_point
        
class TpPadder_1:
    """Collate function for the Trajectory Prediction (TP) task.
    """

    def __init__(self, pred_len, eval=True):
        self.pred_len = pred_len
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)

            input_row = traj_feats[:-self.pred_len]
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)
            input_row = np.concatenate([input_row, repeat(np.array([FEATURE_PAD, MASK_TOKEN]),
                                                          'a -> 1 F a', F=input_row.shape[1])])
            input_batch.append(input_row)

            output_row = traj_feats[-self.pred_len:]
            output_row = np.stack([output_row, np.ones_like(output_row) * UNKNOWN_TOKEN], -1)
            output_batch.append(output_row)

            pos_batch.append(np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1))  # (L, 2)

        input_batch, output_batch, pos_batch = pad_batch_3d(input_batch), \
            pad_batch_3d(output_batch), pad_batch_2d(pos_batch)  # (B, L_in/out, F, 2), (B, L_in, 2)

        input_tensor = torch.from_numpy(
            np.concatenate([input_batch,
                            repeat(np.array([FEATURE_PAD, START_TOKEN]), 'a -> B 1 F a',
                                   B=input_batch.shape[0], F=input_batch.shape[2])], axis=1)).float()
        if not self.eval:
            input_tensor = torch.cat([input_tensor, torch.from_numpy(output_batch).float()], dim=1)
        output_tensor = torch.from_numpy(
            np.concatenate([input_batch, output_batch,
                            repeat(np.array([FEATURE_PAD, END_TOKEN]), 'a -> B 1 F a',
                                   B=input_batch.shape[0], F=input_batch.shape[2])], axis=1)).float()
        pos_tensor = torch.from_numpy(
            np.concatenate([pos_batch,
                            np.stack([repeat(np.max(pos_batch, axis=1)[..., 0], 'B -> B L', L=self.pred_len+1),
                                      repeat(np.arange(1, self.pred_len+2), 'L -> B L', B=input_batch.shape[0])],
                                     axis=-1)], axis=1)).long()

        return input_tensor, output_tensor, pos_tensor


class TrajTtePadder:
    """Collate function for the Trajectory-based TTE Task.
    """

    def __init__(self):
        pass

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            input_row = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)
            output_row = np.copy(input_row)
            pos_row = np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1)

            input_row[1:, ST_MAP['temporal'], 0] = FEATURE_PAD
            input_row[1:, ST_MAP['temporal'], 1] = MASK_TOKEN
            output_row[1:, ST_MAP['temporal'], 1] = UNKNOWN_TOKEN

            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)  # (L, 2)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()
        return input_batch, output_batch, pos_batch


class OdTtePadder:
    """Collate function for the OD-based TTE task.
    """

    def __init__(self, eval=True):
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        first_point = []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            input_row = traj_feats[[0, -1]]
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)
            _UTM = traj[[X_norm_COL, Y_norm_COL]].to_numpy()
            first_point.append(_UTM[0])
            # input_row[1, :, 0] = FEATURE_PAD
            # input_row[1, :, 1] = MASK_TOKEN
            output_row = np.copy(input_row)
            input_row[1, ST_MAP['temporal'], 0] = FEATURE_PAD
            input_row[1, ST_MAP['temporal'], 1] = MASK_TOKEN
            output_row[1, ST_MAP['temporal'], 1] = UNKNOWN_TOKEN
            pos_row = np.arange(input_row.shape[0])

            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_tensor = torch.tensor(pad_batch_2d(pos_batch)).long()
        first_point = torch.tensor(first_point).float()

        return input_batch, output_batch, pos_tensor, first_point


def fetch_task_padder(padder_name, padder_params):
    if padder_name == 'tp':
        task_padder = TpPadder(**padder_params)
    elif padder_name == 'traj_tte':
        task_padder = TrajTtePadder()
    elif padder_name == 'od_tte':
        task_padder = OdTtePadder(**padder_params)
    elif "rec" in padder_name:
        task_padder = TRecPadder(**padder_params)
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')

    return task_padder


def pad_batch_3d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F, 2), (L2, F, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len, batch[0].shape[1]), PAD_TOKEN, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch


def pad_batch_2d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, 2), (L2, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    
    # np.stack((
    #     np.full((len(batch), max_len), FEATURE_PAD, dtype=float),
    #     np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    # ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch