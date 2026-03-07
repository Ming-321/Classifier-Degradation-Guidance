import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CADS:
    """
    CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling
    (https://arxiv.org/abs/2310.17347v4)

    Increases generative diversity by dynamically adding time-step decaying noise to the conditional signal during inference.
    """

    def __init__(
        self,
        pipeline_name: str = "sd3",
        add_noise_scale: float = 0.07,
        tau1: float = 0.6,
        tau2: float = 1.0,
        rescale: bool = False,
        psi: float = 0.5,
        num_train_timesteps: int = 1000,
    ):
        """
        Initializes the CADS sampler.

        Args:
            pipeline_name: Pipeline name, used to set default parameters for specific models.
            add_noise_scale: Noise intensity parameter 's'.
            tau1: First threshold for the annealing schedule function.
            tau2: Second threshold for the annealing schedule function.
            rescale: Whether to enable the rescaling feature.
            psi: Blending parameter for rescaling.
            num_train_timesteps: Number of training timesteps, used for timestep normalization.
        """
        self.pipeline_name = pipeline_name
        self.add_noise_scale = add_noise_scale
        self.tau1 = tau1
        self.tau2 = tau2
        self.rescale = rescale
        self.psi = psi
        self.num_train_timesteps = num_train_timesteps

        # Default parameters for different models
        self._model_configs = {
            "sd3": {"tau1": 0.6, "tau2": 1.0},
            "sd": {"tau1": 0.6, "tau2": 1.0},
            "flux": {"tau1": 0.6, "tau2": 1.0},
        }

        # Apply model-specific configuration
        if pipeline_name in self._model_configs:
            config = self._model_configs[pipeline_name]
            self.tau1 = config.get("tau1", self.tau1)
            self.tau2 = config.get("tau2", self.tau2)

    def linear_schedule(self, t_normalized: float) -> float:
        """
        Linear annealing schedule function.

        Args:
            t_normalized: Normalized timestep (0 corresponds to inference start, 1 to inference end).

        Returns:
            Gamma value (between 0 and 1).
        """
        if t_normalized <= self.tau1:
            return 1.0
        if t_normalized >= self.tau2:
            return 0.0
        gamma = (self.tau2 - t_normalized) / (self.tau2 - self.tau1)
        return gamma

    def add_noise(
        self, y: torch.Tensor, gamma: float, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Adds noise to the conditional vector.

        Args:
            y: Input conditional vector.
            gamma: Gamma value for the current timestep.
            device: Device.

        Returns:
            Conditional vector after adding noise.
        """
        if device is None:
            device = y.device

        # Save original statistics (for rescaling)
        if self.rescale:
            y_mean = y.mean(dim=-1, keepdim=True)
            y_std = y.std(dim=-1, keepdim=True)

        # Generate noise
        noise = torch.randn_like(y, device=device)

        # Apply CADS formula: ŷ = √γ(t) * y + s * √(1-γ(t)) * n
        gamma_sqrt = gamma**0.5
        one_minus_gamma_sqrt = (1 - gamma) ** 0.5

        y_noisy = gamma_sqrt * y + self.add_noise_scale * one_minus_gamma_sqrt * noise

        # Optional rescaling step
        if self.rescale and gamma < 1.0:  # Only rescale if noise was added
            y_noisy_mean = y_noisy.mean(dim=-1, keepdim=True)
            y_noisy_std = y_noisy.std(dim=-1, keepdim=True)

            # Avoid division by zero
            y_noisy_std = torch.where(
                y_noisy_std < 1e-8, torch.ones_like(y_noisy_std), y_noisy_std
            )

            # Rescale to original distribution
            y_scaled = (y_noisy - y_noisy_mean) / y_noisy_std * y_std + y_mean

            # Blend using psi parameter
            y_noisy = self.psi * y_scaled + (1 - self.psi) * y_noisy

        return y_noisy

    def normalize_timestep(
        self, t: torch.Tensor, step_index: int, num_inference_steps: int
    ) -> float:
        """
        Normalizes the scheduler's timestep to a 0-1 range.

        Args:
            t: Current timestep (obtained from the scheduler).
            step_index: Current step index.
            num_inference_steps: Total number of inference steps.

        Returns:
            Normalized timestep.
        """
        # Normalize using step index for stability
        # At inference start, step_index=0 -> t_normalized=0
        # At inference end, step_index=num_inference_steps-1 -> t_normalized close to 1
        t_normalized = step_index / max(num_inference_steps - 1, 1)
        return t_normalized

    def __call__(
        self,
        y_content: torch.Tensor,
        y_null: torch.Tensor,
        t: torch.Tensor,
        step_index: int,
        num_inference_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Main call interface for CADS.

        Args:
            y_content: Conditional vector (positive prompt).
            y_null: Null conditional vector (negative prompt).
            t: Current timestep.
            step_index: Current step index.
            num_inference_steps: Total number of inference steps.

        Returns:
            Processed (y_content_hat, y_null_hat).
        """
        # Normalize timestep
        t_normalized = self.normalize_timestep(t, step_index, num_inference_steps)

        # Calculate gamma value for the current timestep
        gamma = self.linear_schedule(t_normalized)

        # Add noise to content and null conditional vectors separately
        y_content_hat = self.add_noise(y_content, gamma, device=y_content.device)
        y_null_hat = self.add_noise(y_null, gamma, device=y_null.device)

        return y_content_hat, y_null_hat

    def get_config(self) -> Dict[str, Any]:
        """Returns the current configuration."""
        return {
            "pipeline_name": self.pipeline_name,
            "add_noise_scale": self.add_noise_scale,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "rescale": self.rescale,
            "psi": self.psi,
            "num_train_timesteps": self.num_train_timesteps,
        }

    def __repr__(self) -> str:
        return f"CADS(pipeline={self.pipeline_name}, s={self.add_noise_scale}, tau1={self.tau1}, tau2={self.tau2})"
