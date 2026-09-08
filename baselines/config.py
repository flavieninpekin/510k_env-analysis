"""Shared hyperparameter configuration for 510K baselines."""
from dataclasses import dataclass, field, asdict
from typing import Optional


MODES = ["single", "static", "dynamic", "obvious"]


@dataclass
class PPOConfig:
    """Masked PPO hyperparameters for a single training run."""

    mode: str = "single"          # one of MODES
    seed: int = 0
    policy: str = "mlp"           # "mlp" | "lstm" | "ippo"
    opponent: str = "rule"        # "random" | "rule" (auto-play opponents)
    total_timesteps: int = 1_000_000
    chunk_size: int = 100_000     # steps trained per subprocess invocation
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    net_arch: tuple = (256, 256)
    lstm_hidden_size: int = 256
    out_dir: str = "runs"
    eval_episodes: int = 100
    reveal: Optional[float] = None   # info-reveal prob for DYNAMIC (0..1), None=off

    def tag(self) -> str:
        if self.policy == "ippo":
            base = "ippo"
        else:
            base = f"ppo_{self.policy}_{self.opponent}"
        if self.reveal is not None:
            base += f"_r{self.reveal:g}"
        return base

    def to_dict(self) -> dict:
        d = asdict(self)
        d["net_arch"] = list(self.net_arch)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        d = dict(d)
        d["net_arch"] = tuple(d.get("net_arch", (256, 256)))
        return cls(**d)