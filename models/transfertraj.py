import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from models.encode import PositionalEncode, FourierEncode, RAFEE_Encoder
from data import KNOWN_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN, ST_MAP, coord_transform_GPS_UTM
import numpy as np

S_COLS = ST_MAP['spatial']
T_COLS = ST_MAP['temporal']

def load_transfer_feature(model, UTM_region, spatial_middle_coord, poi_embed, poi_coors):
    model.UTM_region = UTM_region
    model.spatial_middle_coord = spatial_middle_coord
    model.poi_embed_mat = poi_embed
    model.poi_coors = poi_coors
    return model

class TransferTraj(nn.Module):
    def __init__(self, embed_size, d_model, poi_embed, poi_coors, road_embed, road_coors, rafee_layer, UTM_region, poi_dist, rn_dist):
        """Trajectory Fundational Model.

        Args:
            embed_size (int): the dimension of learned embedding modules.
            d_model (int): the dimension of the sequential model.
            poi_embed (np.array): pre-defined embedding matrix of all POIs, with shape (n_poi, E).
            poi_coors (np.array): coordinates of all POIs, with shape (n_poi, 2).
            spatial_border (np.array): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
        """
        super().__init__()

        self.poi_coors = poi_coors
        self.road_coors = road_coors
        self.UTM_region = UTM_region
        self.poi_dist = poi_dist
        self.rn_dist = rn_dist

        # Embedding layers for mapping raw features into latent embeddings.
        self.spatial_embed_layer = nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(),
                                                 nn.Linear(embed_size, d_model))
        self.temporal_embed_modules = nn.ModuleList([FourierEncode(embed_size) for _ in range(4)])
        self.temporal_embed_layer = nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model))
        # self.poi_embed_mat = nn.Embedding(*poi_embed.shape)
        # self.poi_embed_mat.weight = nn.Parameter(torch.from_numpy(poi_embed).float(), requires_grad=False)
        self.poi_embed_mat = poi_embed 
        self.road_embed_mat = road_embed

        self.poi_embed_layer = nn.Sequential(nn.LayerNorm(poi_embed.shape[1]),
                                             nn.Linear(poi_embed.shape[1], d_model))

        self.road_embed_layer = nn.Sequential(nn.LayerNorm(road_embed.shape[1]),
                                             nn.Linear(road_embed.shape[1], d_model))

        self.token_embed_layer = nn.Sequential(nn.Embedding(6, embed_size, padding_idx=5), nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, d_model))
        self.pos_encode_layer = PositionalEncode(d_model)

        # Self-attention layer for aggregating the modals.
        self.modal_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=1)

        # Sequential model.
        # self.seq_model = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
        #     num_layers=2)

        self.seq_model = RAFEE_Encoder(d_model, layers=rafee_layer)

        # Prediction modules.
        self.spatial_pred_layer = nn.Sequential(nn.Linear(d_model, 2))
        self.temporal_pred_layer = nn.Sequential(nn.Linear(d_model, 4), nn.Softplus())
        self.token_pred_layers = nn.ModuleList([nn.Linear(d_model, 5) for _ in range(2)])

    def forward(self, input_seq, positions, first_point):
        L = input_seq.size(1)
        # Fetch and embed token modal.
        token = input_seq[..., [S_COLS[0], T_COLS[0]], 1].long()  # (B, L, F)
        # Fetch and embed spatial modal, including POIs.
        spatial = input_seq[:, :, S_COLS, 0]  # (B, L, 2)
        # Fetch and embed temporal modal.
        temporal = input_seq[:, :, T_COLS, 0]  # (B, L, 2)
        temporal_token = tokenize_timestamp(temporal)
        modal_h, norm_coord = self.cal_modal_h(spatial, temporal_token, token, positions, first_point)

        batch_mask = token[..., 0] == PAD_TOKEN
        causal_mask = gen_causal_mask(L).to(input_seq.device)
        
        mem_seq = self.seq_model(modal_h, norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask)
        return modal_h, mem_seq

    def cal_modal_h(self, spatial, temporal_token, token, positions, first_point, K=3):
        """Calculate modal hidden states with the given features.

        Args:
            spatial (Tensor): spatial features with shape (B, L, 2).
            temporal_token (Tensor): temporal tokens with shape (B, L, 4).
            token (Tensor): spatial and temporal tokens with shape (B, L, 2).
            positions (Tensor): dual-layer position indices with shape (B, L, 2).

        Returns:
            Tensor: the sequence of modal hidden states with shape (B, L, E).
        """
        B = spatial.size(0)

        # Embedding of tokens for the spatial and temporal modals.
        token_e = self.token_embed_layer(token)  # (B, L, F, E)

        # Mask used to fill the embedding of features where the features are padding values.
        # Specifically, features where the token is not "KNOWN" or "UNKNOWN".
        feature_e_mask = ~torch.isin(token, torch.tensor([KNOWN_TOKEN, UNKNOWN_TOKEN]).to(token.device))  # (B, L, 2)

        # MinMax normalize spatial coordinates with borders.
        # norm_coord = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
        #     (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        
        # Scale normaliza spatial coordinates with middle point
        # norm_coord = (spatial - self.spatial_middle_coord.unsqueeze(0)) / self.scale  # 在输入前就做归一化
        
        norm_coord = spatial

        spatial_e = self.spatial_embed_layer(norm_coord)  # (B, L, E)
        spatial_e.masked_fill(feature_e_mask[..., 0].unsqueeze(-1), 0)
        spatial_e += token_e[:, :, 0]

        # exit()
        # Calculate nearest K POI and road of each coordinates.
        # poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) -
        #         (spatial + first_point.unsqueeze(1)).unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1)
        poi_distance = ((self.poi_coors.unsqueeze(0).unsqueeze(0) -
                (spatial + first_point.unsqueeze(1)).unsqueeze(2)) ** 2).sum(-1)
        # 构造 mask：选出距离小于 100 的 POI
        poi_mask = poi_distance < self.poi_dist  # (B, T, N_poi)，True 表示该 POI 被选中
        poi_selected = self.poi_embed_layer(self.poi_embed_mat).unsqueeze(0).unsqueeze(0).expand(poi_distance.shape[0], poi_distance.shape[1], -1, -1)  # (B, T, N_poi, D)
        poi_selected = poi_selected * poi_mask.unsqueeze(-1)  # 将不满足条件的置为0
        # 统计每个 (B,T) 位置上有多少个 POI 被选中，用于做平均
        valid_counts = poi_mask.sum(-1, keepdim=True).clamp(min=1)  # 避免除以0
        # 对选中的 POI 嵌入取均值
        poi_e = poi_selected.sum(dim=2) / valid_counts  # (B, T, D)
        # 掩码填充
        poi_e.masked_fill_(feature_e_mask[..., 0].unsqueeze(-1), 0)
        poi_e += token_e[:, :, 0]


        
        road_distance = ((self.road_coors.unsqueeze(0).unsqueeze(0) -
                (spatial + first_point.unsqueeze(1)).unsqueeze(2)) ** 2).sum(-1)
        road_mask = road_distance < self.rn_dist  # (B, T, N_poi)，True 表示该 road network 被选中
        road_selected = self.road_embed_layer(self.road_embed_mat).unsqueeze(0).unsqueeze(0).expand(road_distance.shape[0], road_distance.shape[1], -1, -1)  # (B, T, N_poi, D)
        road_selected = road_selected * road_mask.unsqueeze(-1)  # 将不满足条件的置为0
        valid_counts = road_mask.sum(-1, keepdim=True).clamp(min=1)  # 避免除以0
        road_e = road_selected.sum(dim=2) / valid_counts  # (B, T, D)
        road_e.masked_fill(feature_e_mask[..., 0].unsqueeze(-1), 0)
        road_e += token_e[:, :, 0]


        # Embed temporal tokens.
        temporal_e = torch.cat([module(temporal_token[..., i])
                                for i, module in enumerate(self.temporal_embed_modules)], -1)
        temporal_e = self.temporal_embed_layer(temporal_e)
        temporal_e.masked_fill(feature_e_mask[..., 1].unsqueeze(-1), 0)
        temporal_e += token_e[:, :, 1]

        # Aggregate and mix the hidden states of all modals.
        modal_e = rearrange(torch.stack([spatial_e, temporal_e, poi_e, road_e], 2),
                            'B L F E -> (B L) F E')
                            
        modal_h = rearrange(self.modal_mixer(modal_e), '(B L) F E -> B L F E', B=B).mean(axis=2)

        # Incorporate dual-layer positional encoding.
        pos_encoding = self.pos_encode_layer(positions) #self.pos_encode_layer(positions[..., 0]) + self.pos_encode_layer(positions[..., 1])
        modal_h += pos_encoding

        return modal_h, norm_coord

    def pred(self, mem_seq, return_raw=True):
        """Predict all features given the hidden sequence produced by the sequential model.

        Args:
            mem_seq (Tensor): memory sequence with shape (B, L, E).

        Returns:
            Tensor: predicted spatial coordinates with shape (B, L, 2).
            Tensor: predicted temporal tokens with shape (B, L, 4).
            List: predicted token distributions, each item is a Tensor with shape (B, L, n_token).
        """
        pred_spatial = self.spatial_pred_layer(mem_seq)  # B, F, 2
        # pred_spatial = pred_spatial * self.scale + self.spatial_middle_coord.unsqueeze(0)

        # pred_spatial = pred_spatial * (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0) + \
        #     self.spatial_border[0].unsqueeze(0).unsqueeze(0)

        pred_temporal_token = self.temporal_pred_layer(mem_seq)
        pred_token = [layer(mem_seq) for layer in self.token_pred_layers]  # each (B, L, n_token)
        pred_token = torch.stack(pred_token, 2)
        # if not return_raw:
        #     pred_token = torch.argmax(torch.stack(pred_token, 2), -1)  # (B, L, 2)
        pred_token_max = torch.argmax(pred_token, -1)  # (B, L, 2)


        return pred_spatial, pred_temporal_token, pred_token, pred_token_max

    def loss(self, input_seq, target_seq, positions, first_point, teacher_ratio=0.5):
        """
        The loss value calculation of TrajFM.

        Args:
            target_seq (torch.FloatTensor): contains the generation target features of shape (B, L, F, 2).
        """
        # print(input_seq.shape, target_seq.shape, positions.shape)
        target_spatial = target_seq[..., S_COLS, 0]  # (B, L, 2)
        target_temp_token = tokenize_timestamp(target_seq[:, :, T_COLS, 0])
        target_token = target_seq[..., [S_COLS[0], T_COLS[0]], 1].long()  # (B, L, 2)
        
        batch_mask = target_seq[:, :, 0, 0] == PAD_TOKEN

        feature_mask = target_token != UNKNOWN_TOKEN
        token_mask = target_token == PAD_TOKEN
        
        B, L, _, _ = target_seq.shape
        _, L_in, _, _ = input_seq.shape
        # print(input_seq)
        modal_h, mem_seq = self.forward(input_seq, positions[:,:L_in], first_point)  # (B, L, E)

        pred_spatial, pred_temporal_token, pred_token_dist, pred_token = self.pred(mem_seq)  

        spatial_step, temporal_token_step, token_step = pred_spatial[:, -1:], \
            pred_temporal_token[:, -1:], pred_token[:, -1:]

        # all_norm_coord = input_seq[:, :, S_COLS, 0]  # (B, L, 2)
        # for i in range(L - L_in):
        #     positions_step = positions[:, L_in+i:L_in+i+1]

        #     # 对于输入使用teacher forcing策略
        #     using_truth = np.random.rand() < teacher_ratio
        #     if using_truth:
        #         curr_truth_loc = L_in + i
        #         spatial_step = target_spatial[:, curr_truth_loc: curr_truth_loc+1]
        #         temporal_token_step = target_temp_token[:, curr_truth_loc: curr_truth_loc + 1]
        #         token_step = target_token[:, curr_truth_loc:curr_truth_loc+1]
        #     modal_h_step, norm_coord = self.cal_modal_h(spatial_step, temporal_token_step, token_step, positions_step)  #单步模态编码
        #     modal_h = torch.cat([modal_h, modal_h_step], 1)  # 将预测出来的这个模态信息和之前的模态拼接起来
        #     all_norm_coord = torch.cat([all_norm_coord, norm_coord], 1)  #将下一步预测的坐标拿来，计算RoPE

        #     L_cur = L_in + i + 1
        #     causal_mask = gen_causal_mask(L_cur).to(input_seq.device)
        #     mem_seq = self.seq_model(modal_h, all_norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask[:, :L_cur])
        #     spatial_step, temporal_token_step, pred_token_step, token_step = self.pred(mem_seq[:, -1:], return_raw=False)
            
        #     pred_spatial = torch.cat([pred_spatial, spatial_step], 1)
        #     pred_temporal_token = torch.cat([pred_temporal_token, temporal_token_step], 1)
        #     pred_token = torch.cat([pred_token, token_step], 1)
        #     pred_token_dist = torch.cat([pred_token_dist, pred_token_step], 1)

        # for i in range(L_tgt - L_in):
        #     positions_step = positions[:, L_in+i-1:L_in+i]
        #     modal_h_step, norm_coord = self.cal_modal_h(spatial_step, temporal_token_step, token_step, positions_step)  #不归一化
        #     modal_h = torch.cat([modal_h, modal_h_step], 1)
        #     all_norm_coord = torch.cat([all_norm_coord, norm_coord], 1)

        #     L_cur = L_in + i + 1

        #     causal_mask = gen_causal_mask(L_cur).to(input_seq.device)
        #     mem_seq = self.seq_model(modal_h, all_norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask[:, :L_cur])

        #     spatial_step, temporal_token_step, token_step = self.pred(mem_seq[:, -1:], return_raw=False)

        #     pred_spatial = torch.cat([pred_spatial, spatial_step], 1)
        #     pred_temporal_token = torch.cat([pred_temporal_token, temporal_token_step], 1)
        #     pred_token = torch.cat([pred_token, token_step], 1)

        # print(pred_spatial)
        # print(target_spatial)
        # print(torch.isnan(pred_spatial).any().item(), torch.isnan(target_spatial).any().item() )

        # spatial_loss = geo_distance(pred_spatial, target_spatial)
        # spatial_loss = masked_mean(spatial_loss, feature_mask[..., 0])
        # print(pred_spatial)
        # print(target_spatial)
        # print(target_spatial.shape)
        # exit()
        spatial_loss = F.mse_loss(pred_spatial, target_spatial, reduction='none')
        # print(spatial_loss)
        spatial_loss = masked_mean(spatial_loss, feature_mask[..., 0].unsqueeze(-1))
        
        temporal_loss = F.mse_loss(pred_temporal_token, target_temp_token, reduction='none')
        
        temporal_loss = masked_mean(temporal_loss, feature_mask[..., 1].unsqueeze(-1))
        # print("pred_token_dist:", pred_token_dist.shape, "target_token", target_token.shape)
        token_loss = rearrange(torch.stack([F.cross_entropy(rearrange(pred_token_dist[:,:, i], 'B L N -> (B L) N'),
                                                            rearrange(torch.clamp(target_token[..., i], max=4), 'B L -> (B L)'), reduction='none')
                                            for i in range(2)], -1), '(B L) F -> B L F', B=B)
        token_loss = masked_mean(token_loss, token_mask)
        
        return spatial_loss + temporal_loss + token_loss

    @torch.no_grad()
    def test(self, input_seq, target_seq, positions, first_point):
        """The auto-regressive test process of TrajFM.

        Args:
            input_seq (torch.FloatTensor): contains the input features of shape (B, L_in, F, 2).
            Different from the `input_seq` in `forward`, an extra start step should be included in the end.
            target_seq (torch.FloatTensor): contains the target features of shape (B, L_tgt, F, 2).
            postions (torch.LongTensor): represents the input dual-layer positions of shape (B, L_tgt, 2).

        Returns:
            Tensor: predicted sequence of spatial features with shape (B, L, 2).
            Tensor: predicted sequence of temporal tokens with shape (B, L, 4).
            Tensor: predicted sequence of tokens with shape (B, L, 2).
        """
        B, L_in, L_tgt = input_seq.size(0), input_seq.size(1), target_seq.size(1)

        # print(L_tgt - L_in)
        batch_mask = target_seq[:, :, 0, 0] == PAD_TOKEN
        # print(batch_mask.shape)
        modal_h, mem_seq = self.forward(input_seq, positions[:, :L_in], first_point)
        pred_spatial, pred_temporal_token, _, pred_token = self.pred(mem_seq, return_raw=False)
        spatial_step, temporal_token_step, token_step = pred_spatial[:, -1:], \
            pred_temporal_token[:, -1:], pred_token[:, -1:]
        all_norm_coord = input_seq[:, :, S_COLS, 0]  # (B, L, 2)
        for i in range(L_tgt - L_in):
            positions_step = positions[:, L_in+i : L_in+i+1]
            modal_h_step, norm_coord = self.cal_modal_h(spatial_step, temporal_token_step, token_step, positions_step)  #不归一化
            modal_h = torch.cat([modal_h, modal_h_step], 1)
            all_norm_coord = torch.cat([all_norm_coord, norm_coord], 1)

            L_cur = L_in + i + 1

            causal_mask = gen_causal_mask(L_cur).to(input_seq.device)
            mem_seq = self.seq_model(modal_h, all_norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask[:, :L_cur])

            spatial_step, temporal_token_step, _, token_step = self.pred(mem_seq[:, -1:], return_raw=False)

            pred_spatial = torch.cat([pred_spatial, spatial_step], 1)
            pred_temporal_token = torch.cat([pred_temporal_token, temporal_token_step], 1)
            pred_token = torch.cat([pred_token, token_step], 1)

        target_spatial = target_seq[..., S_COLS, 0]  # (B, L, 2)
        target_temp_token = tokenize_timestamp(target_seq[:, :, T_COLS, 0])
        target_token = target_seq[..., [S_COLS[0], T_COLS[0]], 1].long()  # (B, L, 2)

        
        # 最后将预测的结果和真实的结果反归一化
        # 先反缩放在加上中间坐标
        
        pred_spatial = pred_spatial + first_point.unsqueeze(1)
        target_spatial = target_spatial + first_point.unsqueeze(1)

        pred_spatial = pred_spatial.cpu().numpy()
        target_spatial = target_spatial.cpu().numpy()

        # print(pred_spatial.shape)
        # print(target_spatial.shape)
        # print(first_point.shape)
        # exit()
        # pred_spatial = pred_spatial + 
        pred_spatial = coord_transform_GPS_UTM(pred_spatial, self.UTM_region, origin_coord="utm")
        target_spatial = coord_transform_GPS_UTM(target_spatial, self.UTM_region, origin_coord="utm")

        pred_temporal_token, pred_token = pred_temporal_token.cpu().numpy(), pred_token.cpu().numpy()
        target_temp_token, target_token = target_temp_token.cpu().numpy(), target_token.cpu().numpy()
        # 将它转换为经纬度坐标
        
        return [pred_spatial, pred_temporal_token, pred_token], \
            [target_spatial, target_temp_token, target_token]








def masked_mean(values, mask):
    values = values.masked_fill(mask, 0).sum()
    count = (~mask).long().sum()
    if count == 0: return 0
    return values / count


def gen_causal_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


def tokenize_timestamp(t):
    """Calcualte temporal tokens given the timestamp and delta time.

    Args:
        t (Tensor): raw temporal features with shape (..., 2), the two channels representing 
        the timestamp and time difference with regard to the first point in seconds, respectively.

    Returns:
        Tensor: shape (..., 4) with channels representing the week, hour, minute, 
        and time difference with regard to the first point in minutes, respectively.
    """
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    return torch.stack([week, hour, minute, d_minute], -1)


def geo_distance(a_coor, b_coor):
    """Calculate the geographical distance on Earth's surface.

    Args:
        a_coor (Tensor): one batch of coordiantes with shape (..., 2).
        b_coor (Tensor): another batch of coordinates with shape (..., 2).

    Returns:
        Tensor: Calculated geographical distance in meters with shape (...).
    """
    a_coor, b_coor = torch.deg2rad(a_coor), torch.deg2rad(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = a_x - b_x
    d_y = a_y - b_y

    a = torch.sin(d_y / 2) ** 2 + torch.cos(a_y) * torch.cos(b_y) * torch.sin(d_x / 2) ** 2
    distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000
    return distance


