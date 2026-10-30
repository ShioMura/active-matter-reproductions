"""Small interface check, not an equilibrium or trained-model accuracy test."""
from pathlib import Path
import sys
import tempfile
import random
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "thermomaps-root"))
from ising.base import IsingModel
from ising.samplers import SingleSpinFlipSampler
from ising.observables import Energy, Magnetization
from data.dataset import MultiEnsembleDataset
from data.generic import Summary
from tm.core.prior import GlobalEquilibriumHarmonicPrior
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_process import VPDiffusion
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.architectures.unet_2d_mid_attn import Unet2D


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    trajectories = []
    for temp in (1.5, 2.5):
        model = IsingModel(sampler=SingleSpinFlipSampler, size=8,
                           warmup=10, temp=temp, Jx=1, Jy=1)
        model.simulate(steps=16, observables=[Energy(Jx=1, Jy=1), Magnetization()],
                       sampling_frequency=1)
        frames = np.asarray(model.trajectory.coordinates)
        assert frames.shape == (16, 8, 8)
        assert np.isin(frames, [-1, 1]).all()
        trajectories.append(model.trajectory)
    dataset = MultiEnsembleDataset(trajectories, Summary())
    train, test = dataset.to_TMLoader(train_size=0.8, test_size=0.2,
                                     state_variables=["temperature"])
    prior = GlobalEquilibriumHarmonicPrior(shape=train.data.shape,
        channels_info={"coordinate": [0], "fluctuation": [1]})
    network = Unet2D(dim=16, dim_mults=(1, 2, 4), resnet_block_groups=8, channels=2)
    backbone = ConvBackbone(model=network, data_shape=train.data_dim, target_shape=8,
                            num_dims=4, lr=1e-3, eval_mode="train")
    diffusion = VPDiffusion(num_diffusion_timesteps=100)
    with tempfile.TemporaryDirectory() as directory:
        trainer = DiffusionTrainer(diffusion, backbone, train, prior,
            model_dir=directory, pred_type="noise", test_loader=test)
        trainer.train(1, loss_type="smooth_l1", batch_size=8)
        assert np.isfinite(trainer.train_losses).all()
        assert np.isfinite(trainer.test_losses).all()
        assert list(Path(directory).glob("*.pt")), "Checkpoint was not saved"
        sampler = SteeredDiffusionSampler(diffusion, backbone, train, prior, pred_type="noise")
        sample = sampler.sample_loop(num_samples=2, batch_size=1, temperature=2.0)
        assert tuple(sample.shape) == (2, 2, 8, 8)
        assert torch.isfinite(sample).all()
    print("PASS: simulation, loader, training/backprop, checkpoint save, batched generation.")
    print("This checks execution only; it does not validate equilibrium or prediction accuracy.")


if __name__ == "__main__":
    main()
