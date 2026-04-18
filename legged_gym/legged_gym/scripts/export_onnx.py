"""
Export a trained actor policy from a .pt checkpoint to ONNX format.

Usage:
    python export_onnx.py \
        --checkpoint legged_gym/logs/Pi_Plus_ground_prone/Apr18_10-10-31_pi_prone_training8/model_4500.pt \
        --num_obs 438 \
        --num_actions 22 \
        --output policy.onnx

Does NOT require IsaacGym — only PyTorch and the rsl_rl module.
"""

import argparse
import torch
import torch.nn as nn


# ── minimal ActorCritic re-implementation (matches rsl_rl/modules/actor_critic.py) ──

def get_activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(),
        "selu": nn.SELU(), "lrelu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid(),
    }[name]


def build_actor(num_obs: int, num_actions: int, hidden_dims, activation: str = "elu") -> nn.Sequential:
    act = get_activation(activation)
    layers: list[nn.Module] = [nn.Linear(num_obs, hidden_dims[0]), act]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers += [nn.Linear(hidden_dims[i], num_actions), nn.Tanh()]
        else:
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), get_activation(activation)]
    return nn.Sequential(*layers)


class ActorOnly(nn.Module):
    """Wraps just the actor for clean ONNX export (no critic, no std param)."""
    def __init__(self, actor: nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


# ── main ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model_XXXX.pt")
    parser.add_argument("--output", default="policy.onnx", help="Output .onnx file path")
    parser.add_argument("--num_obs", type=int, default=438, help="Observation dim (73 × 6 = 438)")
    parser.add_argument("--num_actions", type=int, default=22, help="Action dim")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 1. Build actor with same architecture as training
    actor = build_actor(args.num_obs, args.num_actions, args.hidden_dims, args.activation)
    model = ActorOnly(actor).to(args.device)

    # 2. Load weights — checkpoint contains full ActorCritic state_dict,
    #    we only need keys that start with "actor."
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    full_state = ckpt["model_state_dict"]
    actor_state = {k[len("actor."):]: v for k, v in full_state.items() if k.startswith("actor.")}
    model.actor.load_state_dict(actor_state)
    model.eval()
    print(f"Loaded actor weights from {args.checkpoint}")

    # 3. Dummy input: batch=1, obs_dim=438
    dummy_input = torch.zeros(1, args.num_obs, device=args.device)

    # 4. Classic ONNX export (works with PyTorch 1.10+, no onnx package needed)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Saved ONNX model → {args.output}")

    # 5. Quick sanity check
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.output)
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: np.zeros((1, args.num_obs), dtype=np.float32)})
        print(f"ORT sanity check passed — output shape: {out[0].shape}, range: [{out[0].min():.3f}, {out[0].max():.3f}]")
    except ImportError:
        print("onnxruntime not installed — skipping sanity check")


if __name__ == "__main__":
    main()
