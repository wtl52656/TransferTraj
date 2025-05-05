import os
import string
import random
from datetime import datetime

def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def traj_clip(traj, min_length = 5, max_length = 120):
    
#     traj_length = traj.groupby('trip')['trip'].value_counts()
    
#     clip_id_small = traj_length[traj_length < min_length].index.tolist()
#     clip_id_large = traj_length[traj_length > max_length].index.tolist()
#     clip_id = clip_id_small + clip_id_large
#     # indices_to_delete = df[df['B'] == value_to_delete].index

#     # counts = {element: traj['trip'].tolist().count(element) for element in clip_id}
#     traj_filtered = traj[~traj['trip'].isin(clip_id)]
#     traj_filtered = traj_filtered.reset_index(drop=True)

#     return traj_filtered

def traj_clip(traj, min_length = 5, max_length = 120):
    trip_col = "trip"
    trip_counts = traj[trip_col].value_counts()
    valid_trips = trip_counts.loc[(trip_counts >= min_length) & (trip_counts <= max_length)].index
    df_filtered = traj[traj[trip_col].isin(valid_trips)].copy()


    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered