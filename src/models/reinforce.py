import torch


class Reinforce:

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, likelihoods, mask, sample_size, epsilon):
        min_sample_size = mask.sum(dim=1).min().item()
        sample_size = min(sample_size, min_sample_size)

        all_sample_idxs = []
        all_log_prob = []
        for _ in range(self.num_samples):
            sample_idxs, log_prob = self._sample_without_replacement(likelihoods, sample_size, epsilon)
            all_sample_idxs.append(sample_idxs)
            all_log_prob.append(log_prob)

        sample_idxs = torch.stack(all_sample_idxs, dim=1)
        log_prob = torch.stack(all_log_prob, dim=1)
        return sample_idxs, log_prob

    @staticmethod
    def compute_loss(log_prob, reward):
        baseline = torch.mean(reward, dim=1, keepdim=True)
        loss = torch.mean(-(log_prob * ((reward - baseline) / baseline)))
        return loss

    @staticmethod
    def _sample_without_replacement(likelihoods, sample_size, epsilon):
        sample_idxs = torch.full_like(likelihoods[:, :sample_size], fill_value=-1, dtype=torch.long)
        log_probs = torch.zeros_like(likelihoods[:, :sample_size])
        avail_mask = torch.ones_like(likelihoods, dtype=torch.bool)

        B = likelihoods.size(0)
        for i in range(sample_size):
            # Exploration only
            uniform_idxs = torch.multinomial(avail_mask.float(), num_samples=1)
            uniform_probs = 1. / avail_mask.sum(dim=1, keepdim=True)

            # Exploitation only
            likelihoods_masked = likelihoods * avail_mask
            likelihood_probs = likelihoods_masked / likelihoods_masked.sum(dim=1, keepdim=True)
            likelihood_idxs = torch.multinomial(likelihood_probs, num_samples=1)
            likelihood_probs = likelihood_probs.gather(dim=1, index=likelihood_idxs)

            # Explore with p=epsilon
            is_uniform = (torch.rand_like(likelihoods[:, 0]) < epsilon).unsqueeze(dim=1)
            sample_i_idxs = torch.where(is_uniform, uniform_idxs, likelihood_idxs).squeeze(dim=1)
            sample_i_probs = torch.where(is_uniform, uniform_probs, likelihood_probs).squeeze(dim=1)

            sample_idxs[:, i] = sample_i_idxs
            log_probs[:, i] = torch.log(sample_i_probs)
            avail_mask[torch.arange(B), sample_i_idxs] = 0

        log_prob = log_probs.sum(dim=1)
        return sample_idxs, log_prob
