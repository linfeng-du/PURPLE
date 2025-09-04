import torch


def sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int,
    sample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `num_samples` samples from Plackett-Luce distribution defined by `logits`."""
    sample_size = min(sample_size, mask.sum(dim=1).min().item())

    indices = []
    logps = []

    for _ in range(num_samples):
        sample_indices, sample_logp = _sample_without_replacement(logits, mask, sample_size)
        indices.append(sample_indices)
        logps.append(sample_logp)

    indices = torch.stack(indices, dim=1)
    logps = torch.stack(logps, dim=1)
    return indices, logps


def _sample_without_replacement(
    logits: torch.Tensor,
    mask: torch.Tensor,
    sample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a sample of `sample_size` items from Plackett-Luce distribution defined by `logits`."""
    indices = torch.full_like(logits[:, :sample_size], fill_value=-1, dtype=torch.long)
    logps = torch.zeros_like(logits[:, :sample_size], dtype=torch.float)

    # Create a copy of the mask to avoid in-place operations
    mask = mask.clone()
    batch_indices = torch.arange(logits.shape[0], device=logits.device).unsqueeze(dim=1)

    for i in range(sample_size):
        # Sample based on `logits`
        probs = torch.softmax(logits, dim=1)
        likelihood_indices = torch.multinomial(probs, num_samples=1)
        likelihood_probs = probs.gather(dim=1, index=likelihood_indices)

        indices[:, i] = likelihood_indices.squeeze(dim=1)
        logps[:, i] = torch.log(likelihood_probs).squeeze(dim=1)

        mask[batch_indices, likelihood_indices] = 0
        logits = logits.masked_fill(~mask, float('-inf'))

    logp = logps.sum(dim=1)
    return indices, logp


def compute_loss(logps: torch.Tensor, rewards: torch.Tensor, loss: str) -> torch.Tensor:
    """Compute REINFORCE loss with baseline."""
    if loss == 'reinforce':
        loss = -torch.mean(logps * rewards)
    elif loss == 'baseline':
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        loss = -torch.mean(logps * (rewards - mean) / (std + 1e-9))

    return loss
