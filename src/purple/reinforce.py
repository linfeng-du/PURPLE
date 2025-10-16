import torch


def sample(likelihoods: torch.Tensor, num_samples: int, sample_size: int) -> (
    tuple[torch.Tensor, torch.Tensor]
):
    """Generate `num_samples` samples from Plackett-Luce distribution defined by `likelihoods`."""
    valid_cnt = (likelihoods > 0).sum(dim=1).min().item()
    sample_size = min(sample_size, valid_cnt)

    indices = []
    logps = []

    for _ in range(num_samples):
        sample_indices, sample_logp = _sample_without_replacement(likelihoods, sample_size)
        indices.append(sample_indices)
        logps.append(sample_logp)

    indices = torch.stack(indices, dim=1)
    logps = torch.stack(logps, dim=1)
    return indices, logps


def _sample_without_replacement(likelihoods: torch.Tensor, sample_size: int) -> (
    tuple[torch.Tensor, torch.Tensor]
):
    """Generate sample of `sample_size` items from Plackett-Luce distribution defined by `likelihoods`."""
    indices = torch.full_like(likelihoods[:, :sample_size], fill_value=-1, dtype=torch.long)
    logps = torch.zeros_like(likelihoods[:, :sample_size], dtype=torch.float)

    for i in range(sample_size):
        # Sample based on `likelihoods`
        probs = likelihoods / likelihoods.sum(dim=1, keepdim=True)
        item_indices = torch.multinomial(probs, num_samples=1)
        item_probs = probs.gather(dim=1, index=item_indices)

        indices[:, i] = item_indices.squeeze(dim=1)
        logps[:, i] = torch.log(item_probs).squeeze(dim=1)
        likelihoods = likelihoods.scatter(dim=1, index=item_indices, value=0.)

    logp = logps.sum(dim=1)
    return indices, logp


def compute_loss(logps: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Compute REINFORCE loss."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return -torch.mean(logps * (rewards - mean) / (std + 1e-9))
