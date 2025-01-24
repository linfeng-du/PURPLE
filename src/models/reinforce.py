import torch


class Reinforce:

    def sample(
        self,
        likelihoods: torch.Tensor,
        mask: torch.Tensor,
        n_samples: int,
        sample_size: int,
        epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform multiple rounds of sampling without replacement.

        Args:
            likelihoods: torch.Tensor of shape (batch_size, n_items)
                2D tensor where each row contains the likelihoods of items for an example.
            mask: torch.Tensor of shape (batch_size, n_items)
                2D mask indicating the validity of each item where 0 represents padding.
            n_samples: int
                Number of samples drawn from each example.
            sample_size: int
                See self._sample_without_replacement.
            epsilon: float
                See self._sample_without_replacement.

        Returns:
            indices: torch.Tensor of shape (batch_size, n_samples, sample_size)
                Indices of sampled items.
            log_probs: torch.Tensor of shape (batch_size, n_samples)
                Log probability of each sample.
        """
        max_sample_size = mask.sum(dim=1).min().item()
        sample_size = min(sample_size, max_sample_size)

        all_indices = []
        all_log_probs = []

        for _ in range(n_samples):
            indices, log_prob = self._sample_without_replacement(likelihoods, sample_size, epsilon)
            all_indices.append(indices)
            all_log_probs.append(log_prob)

        indices = torch.stack(all_indices, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        return indices, log_probs

    @staticmethod
    def compute_loss(log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute the REINFORCE loss with baseline.

        Args:
            log_probs: torch.Tensor of shape (batch_size, n_samples)
                Log probability of each sample.
            rewards: torch.Tensor of shape (batch_size, n_samples)
                Reward of each sample.

        Returns:
            loss: torch.Tensor of shape ()
                REINFORCE loss with a baseline.
        """
        baseline = torch.mean(rewards, dim=1, keepdim=True)
        loss = torch.mean(-(log_probs * ((rewards - baseline) / baseline)))
        return loss

    @staticmethod
    def _sample_without_replacement(
        likelihoods: torch.Tensor,
        sample_size: int,
        epsilon: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform sampling without replacement based on a batch of likelihoods.

        Args:
            likelihoods: torch.Tensor of shape (batch_size, n_items)
                2D tensor where each row contains the likelihoods of items for an example.
            sample_size: int
                Number of items to sample from each example.
            epsilon: float
                For each item, sample uniformly with a probability of epsilon.

        Returns:
            indices: torch.Tensor of shape (batch_size, sample_size)
                Indices of sampled items.
            log_prob: torch.Tensor of shape (batch_size,)
                Log probability of each sample.
        """
        batch_size = likelihoods.size(dim=0)

        indices = torch.full_like(likelihoods[:, :sample_size], fill_value=-1, dtype=torch.long)
        log_probs = torch.zeros_like(likelihoods[:, :sample_size], dtype=torch.float)
        avail_mask = torch.ones_like(likelihoods, dtype=torch.bool).masked_fill(likelihoods == 0., 0)

        for i in range(sample_size):
            # Sample uniformly
            uniform_indices = torch.multinomial(avail_mask.float(), num_samples=1)
            uniform_probs = 1. / avail_mask.sum(dim=1, keepdim=True)

            # Sample based on likelihoods
            avail_likelihoods = likelihoods.masked_fill(~avail_mask, 0.)
            avail_probs = avail_likelihoods / avail_likelihoods.sum(dim=1, keepdim=True)

            likelihood_indices = torch.multinomial(avail_probs, num_samples=1)
            likelihood_probs = avail_probs.gather(dim=1, index=likelihood_indices)

            # Sample uniformly with a probability of epsilon
            is_uniform = (torch.rand_like(likelihoods[:, 0]) < epsilon).unsqueeze(dim=1)
            indices_col_i = torch.where(is_uniform, uniform_indices, likelihood_indices)
            probs_col_i = torch.where(is_uniform, uniform_probs, likelihood_probs)

            indices[:, i] = indices_col_i.squeeze(dim=1)
            log_probs[:, i] = torch.log(probs_col_i).squeeze(dim=1)
            avail_mask[torch.arange(batch_size), indices_col_i] = 0

        log_prob = log_probs.sum(dim=1)
        return indices, log_prob
