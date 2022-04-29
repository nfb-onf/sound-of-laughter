import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import pickle
import os
import time
import datetime
import timeit
import copy
import soundfile
import matplotlib.pyplot as plt
import torchvision
from msggan.blocks import GenGeneralConvBlock, GenInitialBlock, DisFinalBlock, DisGeneralConvBlock
from msggan.modules import _equalized_conv2d, update_average
from msggan.inversion import output_audio
from tqdm import tqdm

class MSG_GAN:
    """ Unconditional TeacherGAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            use_eql: whether to use the equalized learning rate
            use_ema: whether to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512,
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=torch.device("cpu"), rgb_c=3, normalize_latents=True):
        """ constructor for the class """

        self.gen = Generator(depth, latent_size, use_eql=use_eql, rgb_c=rgb_c).to(device)

        # Parallelize them if required:
        if device == torch.device("cuda"):
            self.gen = nn.DataParallel(self.gen)
            self.dis = Discriminator(depth, latent_size,
                                     use_eql=use_eql, gpu_parallelize=True, rgb_c=rgb_c).to(device)
        else:
            self.dis = Discriminator(depth, latent_size, use_eql=True, rgb_c=rgb_c).to(device)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.device = device
        self.normalize_latents = normalize_latents

        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()

        if type(dis_optim).__name__ == "ExtraAdam":
            loss.backward(retain_graph=True)
            dis_optim.extrapolation()

            dis_optim.zero_grad()
            loss = loss_fn.dis_loss(real_batch, fake_samples)

        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize generator
        gen_optim.zero_grad()

        if type(gen_optim).__name__ == "ExtraAdam":
            loss.backward(retain_graph=True)
            gen_optim.extrapolation()

            gen_optim.zero_grad()
            loss = loss_fn.gen_loss(real_batch, fake_samples)

        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item()

    def train(self, data, gen_optim, dis_optim, loss_fn,
              start=1, num_epochs=12, feedback_factor=100, checkpoint_factor=1,
              data_percentage=100, num_samples=16,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, torch.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, torch.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        fixed_input = self.sample_latents(num_samples)

        # Test write_samples
        self.write_samples(sample_dir, f"initial", fixed_input, data, True)
        
        # create a global time counter
        global_time = time.time()
        global_step = 0

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(tqdm(data), 1):

                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]
                # create a list of downsampled images from the real images:
                images = [images] + [F.avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.depth)]
                images = list(reversed(images))

                # sample some random latent points
                gan_input = self.sample_latents(extracted_batch_size)

                # optimize the generator:
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)
                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                print("Batch: %d  d_loss: %f  g_loss: %f" % (i, dis_loss, gen_loss))
                # provide a loss feedback
                if (i % feedback_factor) == 0 or i == 1:  # Avoid div by 0 error on small training sets
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(global_step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")

                    dis_optim.zero_grad()
                    gen_optim.zero_grad()

                    self.write_samples(sample_dir, f"gen_e{epoch}_b{i}", fixed_input, data)

                # increment the global_step:
                global_step += 1

                if i > limit:
                    break

            # calculate the time required for the epoch
            stop_time = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                torch.save(self.gen.state_dict(), gen_save_file)
                torch.save(self.dis.state_dict(), dis_save_file)
                torch.save(gen_optim.state_dict(), gen_optim_save_file)
                torch.save(dis_optim.state_dict(), dis_optim_save_file)

                self.write_samples(os.path.join(sample_dir, 'progress'), f"gen_e{epoch}", fixed_input, data, save_audio=True)

                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_"
                                                        + str(epoch) + ".pth")
                    torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()

    def sample_latents(self, num_samples):
        # sample some random latent points
        gan_input = torch.randn(
            num_samples, self.latent_size).to(self.device)

        # normalize them if asked
        if self.normalize_latents:
            gan_input = (gan_input / gan_input.norm(dim=-1, keepdim=True)
                            * (self.latent_size ** 0.5))
        return gan_input

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: Tensor(B x C x H x W)
        """
        z = self.sample_latents(num_samples)
        generated = self.gen(z)
        return generated

    def make_grid(self, mels, path):
        """ Save grid of samples """
        grid = torchvision.utils.make_grid(mels, padding=4, nrow=4)[0]
        grid = grid.squeeze(0).cpu().numpy()
        plt.imsave(path, grid)

        return

    def write_samples(self, base_dir, base_str, z, data, save_audio=False):
        # create a grid of samples and save it
        reses = [str(int(np.power(2, dep))) + "_x_" + str(int(np.power(2, dep + 2)))
                    for dep in range(1, self.depth + 1)]

        gen_img_files = [os.path.join(base_dir, res, base_str + ".png")
                            for res in reses]

        gen_mel_files = [os.path.join(base_dir, 'mel', f"{base_str}_smpl{idx}.pkl") for idx in range(8)]

        # Make sure all the required directories exist
        # otherwise make them
        os.makedirs(base_dir, exist_ok=True)
        for gen_img_file in gen_img_files:
            os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)
        os.makedirs(os.path.dirname(gen_mel_files[0]), exist_ok=True)

        if save_audio:
            os.makedirs(os.path.join(base_dir, "audio"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "grids"), exist_ok=True)
        
        with torch.no_grad():       
            output = self.gen(z)
            # Which output in the "minibatch" to output
            out_sample_id=-1
            for o, fn in zip(output, gen_img_files):
                o = o[out_sample_id].squeeze(0).cpu().numpy()
                plt.imsave(fn, o)

            for out_sample_id, fn_mel in enumerate(gen_mel_files):
                mel = output[-1][out_sample_id].squeeze(0).cpu().numpy()
                with open(fn_mel, "wb") as f:
                    pickle.dump(mel, f)

            # Create grid from highest res output
            mels = output[-1][:32]
            grid_path = os.path.join(base_dir, "grids", f"{base_str}.png")
            self.make_grid(mels, grid_path)

            # Save audio sparingly as inversion is expensive
            if save_audio:
                # Only save audio for highest res
                mel = output[-1][-1].squeeze(0).cpu().numpy()
                audio = data.dataset.mel2audio(mel)

                audio_path = os.path.join(base_dir, "audio", f"{base_str}.wav")
                soundfile.write(audio_path, audio, 16000, format='WAV', subtype='FLOAT')



class Generator(nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True, rgb_c=3):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.rgb_c = rgb_c
        self.latent_size = latent_size

        if self.use_eql:
            def to_rgb(in_channels):
                return _equalized_conv2d(in_channels, rgb_c, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return nn.Conv2d(in_channels, rgb_c, (1, 1), bias=True)

        self.layers = nn.ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = nn.ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))

        return outputs

class Discriminator(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512,
                 use_eql=True, gpu_parallelize=False, rgb_c=3):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.rgb_c = rgb_c
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return _equalized_conv2d(rgb_c, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return nn.Conv2d(rgb_c, out_channels, (1, 1), bias=True)

        self.rgb_to_features = nn.ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)

        # create a module list of the other required general convolution blocks
        self.layers = nn.ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = torch.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y
