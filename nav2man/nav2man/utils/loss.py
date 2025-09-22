import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = config['name']
        self.neg_weight = config.get('neg_weight', 1.0)
        self.lam_weight = config.get('lam_weight', 0.0)
        self.lam_dist = config.get('lam_dist', 0.0)
        self.lam_usage = config.get('lam_usage', 0.0)
        self.min_logprob = config.get('min_logprob', 0.0)

    def forward(self, means, covs, weights, target_point, label):
        if self.name == 'mle':
            return self._mle_loss(means, covs, weights, target_point, label)
        elif self.name == 'mle_max':
            return self._mle_max_loss(means, covs, weights, target_point, label)
        elif self.name == 'ce':
            return self._ce_loss(means, covs, weights, target_point, label)
        else:
            raise ValueError(f"Unknown loss name: {self.name}")

    def _mle_loss(self, means, covs, weights, target_point, label):
        B, K, D = means.size()
        target_point = target_point.unsqueeze(1)
        diff = target_point - means
        inv_covs = torch.inverse(covs).to(dtype=torch.float32)
        diff = diff.to(dtype=torch.float32)

        mahalanobis = torch.sum(
            torch.matmul(diff.unsqueeze(2), inv_covs) * diff.unsqueeze(2), dim=-1
        ).squeeze(-1)

        log_det = torch.logdet(covs)
        log_prob_components = -0.5 * mahalanobis - 0.5 * log_det - 0.5 * D * np.log(2 * np.pi)
        log_prob_components = log_prob_components + torch.log(weights + 1e-6)
        
        # Compute responsibilities for usage balancing
        # r_nk = π_k * N(x_n | μ_k, Σ_k) / Σ_j π_j * N(x_n | μ_j, Σ_j)
        max_log_prob_comp = torch.max(log_prob_components, dim=1, keepdim=True)[0]
        log_responsibilities = log_prob_components - torch.logsumexp(log_prob_components, dim=1, keepdim=True)
        responsibilities = torch.exp(log_responsibilities)  # Shape: (B, K)
        
        # Compute final log probability for the mixture
        log_prob = torch.logsumexp(log_prob_components - max_log_prob_comp, dim=1) + max_log_prob_comp.squeeze(1)
        
        if self.min_logprob < 0.0:
            log_prob[(label == -1) & (log_prob < self.min_logprob)] = self.min_logprob

        label = label.float().to(log_prob.device)
        label[label == -1] = -self.neg_weight  # optional handling of ignore index

        entropy_weight = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        entropy_dist = -torch.sum(weights * (0.5 * (D * (1 + math.log(2 * math.pi)) + log_det)), dim=-1).mean()
        
        # Usage balancing term
        # Compute average usage u_k = (1/N) * Σ_n r_nk
        # Only consider samples with positive labels for usage computation
        valid_mask = (label > 0).float().unsqueeze(1)  # Shape: (B, 1)
        valid_responsibilities = responsibilities * valid_mask  # Shape: (B, K)
        num_valid_samples = torch.sum(valid_mask) + 1e-6
        
        avg_usage = torch.sum(valid_responsibilities, dim=0) / num_valid_samples  # Shape: (K,)
        
        # Usage entropy: H(u) = -Σ_k u_k * log(u_k)
        usage_entropy = -torch.sum(avg_usage * torch.log(avg_usage + 1e-6))

        loss = - torch.mean(log_prob * label) - self.lam_weight * entropy_weight - self.lam_dist * entropy_dist - self.lam_usage * usage_entropy
        return loss, entropy_weight, entropy_dist, usage_entropy
    
    def _mle_max_loss(self, means, covs, weights, target_point, label):
        B, K, D = means.size()
        target_point = target_point.unsqueeze(1)
        diff = target_point - means
        inv_covs = torch.inverse(covs).to(dtype=torch.float32)
        diff = diff.to(dtype=torch.float32)

        mahalanobis = torch.sum(
            torch.matmul(diff.unsqueeze(2), inv_covs) * diff.unsqueeze(2), dim=-1
        ).squeeze(-1)

        log_det = torch.logdet(covs)
        log_prob = -0.5 * mahalanobis - 0.5 * log_det - 0.5 * D * np.log(2 * np.pi)
        # log_prob = log_prob + torch.log(weights + 1e-6)
        # max_log_prob = torch.max(log_prob, dim=1, keepdim=True)[0]
        # log_prob = torch.logsumexp(log_prob - max_log_prob, dim=1) + max_log_prob.squeeze(1)
        log_prob, _ = torch.max(log_prob, dim=1)
        if self.min_logprob < 0.0:
            log_prob[(label == -1) & (log_prob < self.min_logprob)] = self.min_logprob

        label = label.float().to(log_prob.device)
        label[label == -1] = -self.neg_weight  # optional handling of ignore index

        entropy_weight = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        entropy_dist = -torch.sum(weights * (0.5 * (D * (1 + math.log(2 * math.pi)) + log_det)), dim=-1).mean()

        loss = - torch.mean(log_prob * label) - self.lam_weight * entropy_weight - self.lam_dist * entropy_dist
        return loss

    def _ce_loss(self, means, covs, weights, target_point, label):
        B, K, D = means.size()
        target_point = target_point.unsqueeze(1)
        diff = target_point - means
        inv_covs = torch.inverse(covs).to(dtype=torch.float32)
        diff = diff.to(dtype=torch.float32)

        mahalanobis = torch.sum(
            torch.matmul(diff.unsqueeze(2), inv_covs) * diff.unsqueeze(2), dim=-1)

        log_det = torch.logdet(covs)
        log_prob = -0.5 * mahalanobis - 0.5 * log_det - 0.5 * D * np.log(2 * np.pi)
        log_prob = log_prob + torch.log(weights + 1e-6)

        max_log_prob = torch.max(log_prob, dim=1, keepdim=True)[0]
        log_prob = torch.logsumexp(log_prob - max_log_prob, dim=1) + max_log_prob.squeeze(1)
        pdf = torch.exp(log_prob)
        success_rate = torch.sigmoid(pdf)

        label = label.float().to(log_prob.device)
        label = (label + torch.abs(label)) / 2

        loss = -torch.mean(
            torch.log(success_rate + 1e-6) * label +
            torch.log(1 - success_rate + 1e-6) * (1 - label)
        )
        return loss