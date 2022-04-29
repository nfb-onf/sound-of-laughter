# sound-of-laughter
L’éclat du rire (The Sound of Laughter) - AI techniques for detecting and generating laughter

This repository contains:
- A MSGGAN designed to work on mel spectrograms
- A simple dataloader designed to load data from a folder of `.wav` files.
- A training script for the MSGGAN

## Training MSGGAN

If required, install the dependencies (in a venv with and upgraded `pip`)
```
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
```

Set the path to the base dir of the project  
```
$ export PYTHONPATH=$PYTHONPATH:`pwd`
```

Run the training script  
```
$ python msggan/train_msggan.py \
--data_path <data_root> \
--root_path <output_path> \
--epochs 500
```

