# TransferTraj:  Vehicle Trajectory Learning Model for Region and Task Transferability

## Hands-on

Install requirements:

```bash
pip install -r requirements.txt
```

Also, you need to install PyTorch. It is a bit more complicated depending on your compute platform.

Set OS env parameters:

```bash
export SETTINGS_CACHE_DIR=/dir/to/cache/setting/files;
export MODEL_CACHE_DIR=/dir/to/cache/model/parameters;
export PRED_SAVE_DIR=/dir/to/save/predictions;
export LOG_SAVE_DIR=/dir/to/save/logs;
```

Run the main script:

```bash
python main.py -s local_test;
```
