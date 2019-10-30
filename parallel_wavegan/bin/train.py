#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Parallel WaveGAN."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.optimizers import RAdam
from parallel_wavegan.utils.dataset import PyTorchDataset


class Trainer(object):
    """Customized trainer module."""

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
        """Initialize trainer."""
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
        self.total_loss = {}

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         ascii=True)
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("finished training.")

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        batch = [b.to(self.device) for b in batch]
        z, c, y, _ = batch

        # calculate loss for generator
        loss = {}
        y_ = self.model["generator"](z, c)
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_.unsqueeze(1)).squeeze(1)
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            gen_loss = sc_loss + mag_loss + self.config["lambda_adv"] * adv_loss
            loss.update({
                "spectral_convergenge_loss": sc_loss.item(),
                "log_stft_magnitude_loss": mag_loss.item(),
                "adversarial_loss": adv_loss.item(),
                "generator_loss": gen_loss.item(),
            })
        else:
            gen_loss = sc_loss + mag_loss
            loss.update({
                "spectral_convergenge_loss": sc_loss.item(),
                "log_stft_magnitude_loss": mag_loss.item(),
                "generator_loss": gen_loss.item(),
            })

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        if self.steps > self.config["discriminator_train_start_steps"]:
            # calculate discriminator loss
            p = self.model["discriminator"](y.unsqueeze(1)).squeeze(1)
            p_ = self.model["discriminator"](y_.unsqueeze(1).detach()).squeeze(1)
            dis_loss = self.criterion["mse"](p, p.new_ones(p.size())) + \
                self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            loss["discriminator_loss"] = dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1

        return loss

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            loss = self._train_step(batch)

            # record
            for key, value in loss.items():
                if key in self.total_loss.keys():
                    self.total_loss[key] += value
                else:
                    self.total_loss[key] = value

            # check interval
            self.total_loss = self._check_log_interval(self.total_loss)
            self._check_eval_interval()
            self._check_save_interval()
            self._check_train_finish()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(steps: {self.steps}) finished {self.epochs} epoch training.")
        logging.info(f"training steps per epoch = f{self.train_steps_per_epoch}.")

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
        dis_loss = self.criterion["mse"](p, p.new_ones(p.size())) + \
            self.criterion["mse"](p_, p_.new_zeros(p_.size()))

        # store into dict
        loss = {
            "validation/adversarial_loss": adv_loss.item(),
            "validation/spectral_convergenge_loss": sc_loss.item(),
            "validation/log_stft_magnitude_loss": mag_loss.item(),
            "validation/generator_loss": gen_loss.item(),
            "validation/discriminator_loss": dis_loss.item(),
        }

        return loss

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(step: {self.steps}) start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        eval_steps_per_epoch = 0
        for batch in tqdm(self.data_loader["dev"], ascii=True):
            loss = self._eval_step(batch)
            if eval_steps_per_epoch == 0:
                total_loss = loss
            else:
                for key, value in loss.items():
                    total_loss[key] += value
            eval_steps_per_epoch += 1
        self.eval_steps_per_epoch = eval_steps_per_epoch
        logging.info(f"(step: {self.steps}) finished evaluation.")
        logging.info(f"evaluation steps per epoch = {self.eval_steps_per_epoch}.")

        # save intermediate result
        self._genearete_and_save_intermediate_result(batch)

        # average loss
        for key in total_loss.keys():
            total_loss[key] /= eval_steps_per_epoch
            logging.info(f"(steps: {self.steps}) {key} = {total_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(total_loss)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        with torch.no_grad():
            batch = [b.to(self.device) for b in batch]
            z, c, y, _ = batch
            y_ = self.model["generator"](z, c)
            y, y_ = y[0].view(-1).cpu().numpy(), y_[0].view(-1).cpu().numpy()

        # plot figure and save it
        figname = os.path.join(self.config["outdir"], f"predictions/{self.steps}-steps.png")
        dirname = os.path.dirname(figname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
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

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"),
                self.model, self.optimizer, self.scheduler, self.steps, self.epochs)
            logging.info(f"saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self, loss):
        if self.steps % self.config["log_interval_steps"] == 0:
            self.tqdm.update(self.config["log_interval_steps"])
            for key in loss.keys():
                loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(steps: {self.steps}) {key} = {loss[key]:.4f}.")
            self._write_to_tensorboard(loss)
            return {}
        else:
            return loss

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2
                 ):
        """Initialize customized collater."""
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
            Tensor: Auxiliary feature batch (B, C, T").
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

        # Make padded target signale batch
        xlens = [len(b[0]) for b in batch]
        max_olen = max(xlens)
        y_batch = np.array([self._pad_2darray(b[0].reshape(-1, 1), max_olen) for b in batch], dtype=np.float32)
        y_batch = torch.FloatTensor(y_batch).transpose(2, 1)

        # Make padded conditional auxiliary feature batch
        clens = [len(b[1]) for b in batch]
        max_clen = max(clens)
        c_batch = np.array([self._pad_2darray(b[1], max_clen) for b in batch], dtype=np.float32)
        c_batch = torch.FloatTensor(c_batch).transpose(2, 1)

        # Make input noise signale batch
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


def save_checkpoint(checkpoint_name,
                    model,
                    optimizer,
                    scheduler,
                    steps,
                    epochs):
    """Save states as checkpoint."""
    state_dict = {
        "model": {
            "generator": model["generator"].state_dict(),
            "discriminator": model["discriminator"].state_dict(),
        },
        "optimizer": {
            "generator": optimizer["generator"].state_dict(),
            "discriminator": optimizer["discriminator"].state_dict(),
        },
        "scheduler": {
            "generator": scheduler["generator"].state_dict(),
            "discriminator": scheduler["discriminator"].state_dict(),
        },
        "steps": steps,
        "epochs": epochs,
    }
    if not os.path.exists(os.path.dirname(checkpoint_name)):
        os.makedirs(os.path.dirname(checkpoint_name))
    torch.save(state_dict, checkpoint_name)


def resume_from_checkpoint(checkpoint_name, trainer):
    """Resume from checkpoint."""
    state_dict = torch.load(checkpoint_name)
    trainer.steps = state_dict["steps"]
    trainer.epochs = state_dict["epochs"]
    trainer.model["generator"].load_state_dict(state_dict["model"]["generator"])
    trainer.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
    trainer.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
    trainer.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
    trainer.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
    trainer.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])
    return trainer


def main():
    """Run main process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dumpdir", default=None, type=str,
                        help="Directory including trainning data.")
    parser.add_argument("--dev-dumpdir", default=None, type=str,
                        help="Direcotry including development data.")
    parser.add_argument("--outdir", default=None, type=str,
                        help="Direcotry to save checkpoints.")
    parser.add_argument("--resume", default=None, type=str,
                        help="Checkpoint file path to resume training.")
    parser.add_argument("--config", default="hparam.yml", type=str,
                        help="Yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level (higher is more logging)")
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

    # check direcotry existence
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
        audio_length_threshold = mel_length_threshold * config["hop_size"]
    else:
        mel_length_threshold = None
        audio_length_threshold = None
    dataset = {
        "train": PyTorchDataset(
            dump_root=args.train_dumpdir,
            audio_length_threshold=audio_length_threshold,
            mel_length_threshold=mel_length_threshold),
        "dev": PyTorchDataset(
            dump_root=args.dev_dumpdir,
            audio_length_threshold=audio_length_threshold,
            mel_length_threshold=mel_length_threshold),
    }

    # get data loader
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    collate_fn = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
        aux_context_window=config["generator_params"]["aux_context_window"],
    )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=True,
            collate_fn=collate_fn,
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
    if args.resume is not None:
        trainer = resume_from_checkpoint(args.resume, trainer)
        logging.info(f"resumed from {args.resume}.")

    # run training loop
    trainer.run()


if __name__ == "__main__":
    main()
