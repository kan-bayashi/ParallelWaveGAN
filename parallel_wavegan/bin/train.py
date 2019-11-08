#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import argparse
import logging
import os

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from parallel_wavegan.datasets import AudioMelDataset
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.optimizers import RAdam
from parallel_wavegan.utils import read_hdf5

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "model": {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            },
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.

        """
        state_dict = torch.load(checkpoint_path)
        self.steps = state_dict["steps"]
        self.epochs = state_dict["epochs"]
        self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
        self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
        self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
        self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        batch = [b.to(self.device) for b in batch]
        z, c, y, _ = batch

        # calculate loss for generator
        y_ = self.model["generator"](z, c)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        gen_loss = sc_loss + mag_loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_.unsqueeze(1)).squeeze(1)
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss += self.config["lambda_adv"] * adv_loss
            self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        if self.steps > self.config["discriminator_train_start_steps"]:
            # calculate discriminator loss
            p = self.model["discriminator"](y.unsqueeze(1)).squeeze(1)
            p_ = self.model["discriminator"](y_.unsqueeze(1).detach()).squeeze(1)
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()
            self._check_train_finish()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(steps: {self.steps}) finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        batch = [b.to(self.device) for b in batch]
        z, c, y, _ = batch

        # calculate generator loss
        y_ = self.model["generator"](z, c)
        p_ = self.model["discriminator"](y_)
        y, y_, p_ = y.squeeze(1), y_.squeeze(1), p_.squeeze(1)
        adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        aux_loss = sc_loss + mag_loss
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        # train discriminator
        p = self.model["discriminator"](y.unsqueeze(1)).squeeze(1)
        p_ = self.model["discriminator"](y_.unsqueeze(1)).squeeze(1)
        real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
        fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(step: {self.steps}) start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            with torch.no_grad():
                self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(step: {self.steps}) finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        with torch.no_grad():
            batch = [b.to(self.device) for b in batch]
            z_batch, c_batch, y_batch, _ = batch
            y_batch_ = self.model["generator"](z_batch, c_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size
            Tensor: Target signal batch (B, 1, T).
            LongTensor: Input length batch (B,)

        """
        # Time resolution adjustment
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            self._assert_ready_for_upsampling(x, c, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                x = x[start_step: start_step + self.batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + self.batch_max_frames]
                self._assert_ready_for_upsampling(x, c, self.hop_size, self.aux_context_window)
            else:
                logging.warn(f"removed short sample from batch (length={len(x)}).")
                continue
            new_batch.append((x, c))
        batch = new_batch

        # Make padded target signal batch
        xlens = [len(b[0]) for b in batch]
        max_olen = max(xlens)
        y_batch = np.array([self._pad_2darray(b[0].reshape(-1, 1), max_olen) for b in batch], dtype=np.float32)
        y_batch = torch.FloatTensor(y_batch).transpose(2, 1)

        # Make padded conditional auxiliary feature batch
        clens = [len(b[1]) for b in batch]
        max_clen = max(clens)
        c_batch = np.array([self._pad_2darray(b[1], max_clen) for b in batch], dtype=np.float32)
        c_batch = torch.FloatTensor(c_batch).transpose(2, 1)

        # Make input noise signal batch
        z_batch = torch.randn(y_batch.size())

        # Make the list of the length of input signals
        input_lengths = torch.LongTensor(xlens)

        return z_batch, c_batch, y_batch, input_lengths

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        assert len(x) == (len(c) - 2 * context_window) * hop_size

    @staticmethod
    def _pad_2darray(x, max_len, b_pad=0, constant_values=0):
        return np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
                      mode="constant", constant_values=constant_values)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py).")
    parser.add_argument("--train-dumpdir", type=str, required=True,
                        help="directory including trainning data.")
    parser.add_argument("--dev-dumpdir", type=str, required=True,
                        help="directory including development data.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('skip DEBUG/INFO messages')

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["batch_max_steps"] // config["hop_size"] + \
            2 * config["generator_params"]["aux_context_window"]
    else:
        mel_length_threshold = None
    if config["format"] == "hdf5":
        audio_query, mel_query = "*.h5", "*.h5"
        audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
        mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
    elif config["format"] == "npy":
        audio_query, mel_query = "*-wave.npy", "*-feats.npy"
        audio_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("support only hdf5 or npy format.")
    dataset = {
        "train": AudioMelDataset(
            root_dir=args.train_dumpdir,
            audio_query=audio_query,
            mel_query=mel_query,
            audio_load_fn=audio_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibilty
        ),
        "dev": AudioMelDataset(
            root_dir=args.dev_dumpdir,
            audio_query=audio_query,
            mel_query=mel_query,
            audio_load_fn=audio_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibilty
        ),
    }

    # get data loader
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    collater = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
        aux_context_window=config["generator_params"]["aux_context_window"],
    )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]),
    }

    # define models and optimizers
    model = {
        "generator": ParallelWaveGANGenerator(
            **config["generator_params"]).to(device),
        "discriminator": ParallelWaveGANDiscriminator(
            **config["discriminator_params"]).to(device),
    }
    criterion = {
        "stft": MultiResolutionSTFTLoss(
            **config["stft_loss_params"]).to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    optimizer = {
        "generator": RAdam(
            model["generator"].parameters(),
            **config["generator_optimizer_params"]),
        "discriminator": RAdam(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"]),
    }
    scheduler = {
        "generator": torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"]),
        "discriminator": torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"]),
    }
    logging.info(model["generator"])
    logging.info(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
