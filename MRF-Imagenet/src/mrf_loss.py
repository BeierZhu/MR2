import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MRFLoss(nn.Module):
    def __init__(self, num_classes=1000, lambda_rep=0.1, ema_decay=0.999, device='cuda'):
        super(MRFLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_rep = lambda_rep
        self.ema_decay = ema_decay
        self.device = device
        self.l1norm=5
        
        # Initialize EMA for mean norm and variance
        self.register_buffer('mean_norm_ema', torch.zeros(num_classes))
        self.register_buffer('variance_ema', torch.zeros(num_classes))
        # self.register_buffer('class_counts', torch.tensor(cls_num_list, dtype=torch.float))
        
    def update_ema(self, features, targets):
        """Update EMA of mean norm and variance for each class."""
        with torch.no_grad():
            batch_size = features.size(0)
            for cls in range(self.num_classes):
                cls_mask = (targets == cls)
                cls_features = features[cls_mask]
                if cls_features.size(0) > 0:
                    # Compute per-class mean
                    cls_mean = cls_features.mean(dim=0)
                    mean_norm = cls_mean.norm(self.l1norm).pow(2)
                    
                    # Compute mean squared deviation
                    centered = cls_features - cls_mean
                    variance = centered.norm(self.l1norm, dim=1).pow(2).mean()
                    
                    # Update EMA
                    if torch.isnan(self.mean_norm_ema[cls]) or self.mean_norm_ema[cls] == 0:
                        self.mean_norm_ema[cls] = mean_norm
                        self.variance_ema[cls] = variance
                    else:
                        self.mean_norm_ema[cls] = (self.ema_decay * self.mean_norm_ema[cls] +
                                                (1 - self.ema_decay) * mean_norm)
                        self.variance_ema[cls] = (self.ema_decay * self.variance_ema[cls] +
                                                (1 - self.ema_decay) * variance)
    
    def compute_gamma(self):
        """Compute per-class margins gamma_k."""
        sum_stats = (self.mean_norm_ema + self.variance_ema).pow(1/3)
        gamma = self.num_classes * sum_stats / sum_stats.sum()
        return gamma.to(self.device)
   
    def forward(self, logits, features, targets):
        """Compute combined loss: logit margin CE + representation margin."""
        # Update EMA statistics
        self.update_ema(features, targets)
        self.device=features.device
        
        # Logit margin cross-entropy loss
        gamma = self.compute_gamma()
        scaled_logits = logits / gamma[targets].view(-1, 1)
        ce_loss = nn.CrossEntropyLoss()(scaled_logits, targets)
        
        # Representation margin loss
        bar_s = self.variance_ema.mean()
        rep_loss = 0
        dim_scale = features.shape[-1]
        for cls in range(self.num_classes):
            cls_mask = (targets == cls)
            cls_features = features[cls_mask]
            if cls_features.size(0) > 1:
                # Compute pairwise distances within the class
                pairwise_diff = cls_features.unsqueeze(1) - cls_features.unsqueeze(0)
                pairwise_dist = pairwise_diff.norm(2, dim=2).pow(2)
                # Mask out self-pairs
                eye = torch.eye(cls_features.size(0), device=self.device, dtype=torch.bool)
                pairwise_dist = pairwise_dist.masked_fill(eye, 0)
                # Compute loss terms
                terms = torch.exp((pairwise_dist - 2 * bar_s)/(10*dim_scale))
                terms = terms.masked_fill(eye, 0).sum(dim=1)
                cls_rep_loss = torch.log1p(terms).mean()
                rep_loss += cls_rep_loss
        rep_loss = rep_loss * dim_scale  /10 / self.num_classes if rep_loss != 0 else torch.tensor(0.0, device=self.device)
        
        return ce_loss + self.lambda_rep * rep_loss
        # return self.lambda_rep * rep_loss