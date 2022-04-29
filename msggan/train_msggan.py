import os
import torch.utils.data as tdata
import torch
from laughter_loaders.datasets import WAVDataset, MELDataset, PadCollate

from msggan.models import MSG_GAN
from msggan.loss import RelativisticAverageHingeGAN
import argparse
import numpy as np
from laughter_loaders.transforms import RandomTransform, ChangeSpeed, ChangePitch, Shifting, AddNoise

import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "/network/datasets/restricted/icmlexvo2022_users/icmlexvo2022.var/icmlexvo2022_extract/wav", type=str)
    parser.add_argument('--root_path', default = "/network/scratch/m/marco.jiralerspong/experiments/icml_sound/default", type=str)
    parser.add_argument('--epochs', default = 500, type=int)

    args = parser.parse_args()

    DATA_DIR = args.data_path
    ROOT_DIR = args.root_path
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'models')
    SAMPLES_DIR = os.path.join(ROOT_DIR, 'samples')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    CHECKPOINT = True
    EMA = True
    BATCH_SIZE = 32
    EPOCHS = args.epochs
    G_LR = 0.003
    D_LR = 0.0003

    torch.manual_seed(42)

    # Data augmentation
    transf = RandomTransform([ChangeSpeed(), ChangePitch(), Shifting(shift_direction="right"), AddNoise()])

    dataset = WAVDataset(DATA_DIR, transforms=transf)
    dataset = MELDataset(dataset)
    dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=PadCollate(dim=2, padval=np.log(1e-6)/2.0))

    msg_gan = MSG_GAN(depth=7, device=torch.device('cuda'), rgb_c=1)

    gen_optim = torch.optim.Adam(msg_gan.gen.parameters(), G_LR, [0, 0.99])
    dis_optim = torch.optim.Adam(msg_gan.dis.parameters(), D_LR, [0, 0.99])

    loss_fn = RelativisticAverageHingeGAN(msg_gan.dis)

    epochs = [-1]
    if CHECKPOINT:
        print(f"CHECKING FOR EXISTING CHECKPOINT IN {CHECKPOINT_DIR}")
        list_models = os.listdir(CHECKPOINT_DIR)
        epochs = [-1]
        for fn in list_models:
            epochs.append(int(fn.split('_')[-1].split('.')[0]))

    epoch = max(epochs)
    if CHECKPOINT and epoch>=0:
        print(f'FOUND MAX EPOCH {epoch}')
        print('USING CHECKPOINT')

        gen_save_file = os.path.join(CHECKPOINT_DIR, "GAN_GEN_" + str(epoch) + ".pth")
        dis_save_file = os.path.join(CHECKPOINT_DIR, "GAN_DIS_" + str(epoch) + ".pth")
        gen_optim_save_file = os.path.join(CHECKPOINT_DIR,
                                           "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
        dis_optim_save_file = os.path.join(CHECKPOINT_DIR,
                                           "GAN_DIS_OPTIM_" + str(epoch) + ".pth")


        print('LOADING CHECKPOINTS')
        msg_gan.dis.load_state_dict(torch.load(dis_save_file))
        msg_gan.gen.load_state_dict(torch.load(gen_save_file))
        dis_optim.load_state_dict(torch.load(dis_optim_save_file))
        gen_optim.load_state_dict(torch.load(gen_optim_save_file))
        if EMA:
            gen_shadow_save_file = os.path.join(CHECKPOINT_DIR, "GAN_GEN_SHADOW_" + str(epoch) + ".pth")
            msg_gan.gen_shadow.load_state_dict(torch.load(gen_shadow_save_file))
    else:
        print('NO CHECKPOINT')
        epoch = 0

    msg_gan.train(dataloader, gen_optim, dis_optim, loss_fn, feedback_factor=50, sample_dir=SAMPLES_DIR,
                  save_dir=CHECKPOINT_DIR, num_epochs=EPOCHS, start=epoch + 1, checkpoint_factor=5)
