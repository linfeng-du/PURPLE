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
        """Perform multiple rounds of sampling without replacement based on item likelihoods.

        Args:
            likelihoods (torch.Tensor): Likelihoods of items. Shape (batch_size, n_items)
            mask (torch.Tensor): Validity of items. Shape (batch_size, n_items)
            n_samples (int): Number of samples to draw.
            sample_size (int): Number of items in each sample.
            epsilon (float): Probability of sampling uniformly among all items.

        Returns:
            indices (torch.Tensor):
                Indices of items in the samples. Shape (batch_size, n_samples, sample_size)
            log_probs (torch.Tensor):
                Log probabilities of the samples. Shape (batch_size, n_samples)
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
        """Compute the REINFORCE loss with an average baseline.

        Args:
            log_probs (torch.Tensor):
                Log probabilities of the samples. Shape (batch_size, n_samples)
            rewards (torch.Tensor):
                Rewards associated with the samples. Shape (batch_size, n_samples)

        Returns:
            loss (torch.Tensor):
                Scalar tensor representing the REINFORCE loss with the average baseline. Shape: ()
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
        """Perform sampling without replacement based on item likelihoods.

        Args:
            likelihoods (torch.Tensor): Likelihoods of items. Shape (batch_size, n_items)
            sample_size (int): Number of items in the sample.
            epsilon (float): Probability of sampling uniformly among all items.

        Returns:
            indices (torch.Tensor): Indices of items in the sample. Shape (batch_size, sample_size)
            log_prob (torch.Tensor): Log probability of the sample. Shape (batch_size,)
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
