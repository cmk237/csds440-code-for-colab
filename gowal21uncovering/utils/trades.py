import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def _resnet_logits_features(model, x):
    if not isinstance(model, torch.nn.DataParallel):
        raise TypeError('Hard-pair weighting requires a DataParallel ResNet model.')
    container = model.module
    backbone = container[-1]
    if not hasattr(backbone, 'forward_features') or not hasattr(backbone, 'linear'):
        raise TypeError('Hard-pair weighting supports only ResNet-family models.')
    features = backbone.forward_features(container[0](x) if len(container) > 1 else x)
    return backbone.linear(features), features, backbone.linear.weight

def _hard_pair_nce(logits, features, classifier_weight, targets, mask, offset):
    normalized_features = F.normalize(features, dim=1)
    normalized_weights = F.normalize(classifier_weight, dim=1)
    cosine = normalized_features @ normalized_weights.t()
    false_mask = torch.ones_like(logits, dtype=torch.bool)
    false_mask.scatter_(1, targets.unsqueeze(1), False)
    pair_weight = torch.clamp(cosine + offset, min=1e-6)
    weighted_false_logits = logits + pair_weight.log()
    weighted_false_logits = weighted_false_logits.masked_fill(~false_mask, -torch.inf)
    denominator = torch.logsumexp(
        torch.cat((logits.gather(1, targets.unsqueeze(1)), weighted_false_logits), dim=1),
        dim=1,
    )
    pairwise_loss = denominator - logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.where(mask, pairwise_loss, F.cross_entropy(logits, targets, reduction='none'))


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                attack='linf-pgd', label_smoothing=0.1, sample_weights=None,
                auxiliary_mask=None, hard_pair_clean_weight=False, hard_pair_beta=0.2,
                hard_pair_consistency_weight=0.2):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='none')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    p_natural = F.softmax(model(x_natural), dim=1)
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1), p_natural
                ).sum()
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = delta.detach().requires_grad_(True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), p_natural
                ).sum()
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = (x_natural + delta).detach()
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    if hard_pair_clean_weight:
        _, clean_features, classifier_weight = _resnet_logits_features(model, x_natural)
        clean_mask = ~auxiliary_mask if auxiliary_mask is not None else torch.ones_like(y, dtype=torch.bool)
        loss_natural = _hard_pair_nce(
            logits_natural, clean_features, classifier_weight, y, clean_mask, hard_pair_beta
        ).mean()
    else:
        loss_natural = criterion_ce(logits_natural, y)
    robust_terms = criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_natural, dim=1),
    ).sum(dim=1)
    if sample_weights is not None:
        robust_terms = robust_terms * sample_weights.square()
    loss_robust = robust_terms.sum() / batch_size
    if hard_pair_clean_weight:
        consistency = (
            F.log_softmax(logits_natural, dim=1)
            - F.log_softmax(logits_adv, dim=1)
        ).square().mean(dim=1).mean()
        loss = loss_natural + beta * loss_robust + hard_pair_consistency_weight * consistency
    else:
        loss = loss_natural + beta * loss_robust
    
    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics
