import math
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm.auto import tqdm


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        device,
        *,
        train_batch_size=256,
        train_lr=1e-3,
        weight_decay=0.0,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        save_every=10000,
        results_folder=None,
    ):
        super().__init__()

        assert results_folder is not None, "must specify results folder"
        self.diffusion_model = diffusion_model

        self.device = device
        self.num_samples = 25
        self.save_every = save_every
        self.sample_every = sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.ds = dataset
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )

        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

        # step counter state
        self.step = 0

    def save(self, milestone):
        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"saving model to {ckpt_path}.")
        data = {
            "step": self.step,
            "model": self.diffusion_model.state_dict(),
            "opt": self.opt.state_dict(),
        }

        torch.save(data, ckpt_path)

    def load(self, milestone):

        ckpt_path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        print(f"loading model from {ckpt_path}.")
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        self.diffusion_model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        # Move model and optimizer to the same device
        device = self.device
        self.diffusion_model.to(device)
        for state in self.opt.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def train(self):
        device = self.device
        self.diffusion_model.to(device)

        all_losses = []

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                data, model_kwargs = next(self.dl)
                data = data.to(device)
                model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)

                self.opt.zero_grad()
                loss = self.diffusion_model.p_losses(data, model_kwargs=model_kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.opt.step()

                pbar.set_description(f"loss: {loss.item():.4f}")
                all_losses.append(loss.item())

                self.step += 1

                if self.step % self.save_every == 0:
                    self.save(self.step)

                if self.step % self.sample_every == 0:
                    self.diffusion_model.eval()

                    with torch.no_grad():
                        model_kwargs = self.ds.random_model_kwargs(self.num_samples)
                        model_kwargs["text_emb"] = model_kwargs["text_emb"].to(device)

                        all_images = self.diffusion_model.sample(
                            batch_size=self.num_samples, model_kwargs=model_kwargs
                        )

                    save_image(
                        all_images,
                        os.path.join(self.results_folder, f"sample-{self.step}.png"),
                        nrow=int(math.sqrt(self.num_samples)),
                    )

                pbar.update(1)

        return all_losses
