"""
Loss functions for CMVec2Vec training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from cm_vec2vec.utils import get_device


def reconstruction_loss(
    reconstructions: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute reconstruction loss with proper scaling for normalized vectors.
    """
    total_loss = 0.0
    count = 0

    for key in reconstructions.keys():
        if key in inputs:
            # Use cosine similarity loss instead of MSE for normalized vectors
            rec_norm = F.normalize(reconstructions[key], p=2, dim=1)
            inp_norm = F.normalize(inputs[key], p=2, dim=1)

            # Cosine similarity loss (1 - cosine_similarity)
            cosine_sim = torch.sum(rec_norm * inp_norm, dim=1)
            loss = torch.mean(1.0 - cosine_sim)

            total_loss += loss
            count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def semantic_reconstruction_loss(
    reconstructions: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Semantic reconstruction loss using contrastive learning.
    """
    total_loss = 0.0
    count = 0

    for key in reconstructions.keys():
        if key in inputs:
            # Normalize embeddings
            rec_norm = F.normalize(reconstructions[key], p=2, dim=1)
            inp_norm = F.normalize(inputs[key], p=2, dim=1)

            # Compute similarity matrix
            similarity_matrix = torch.mm(rec_norm, inp_norm.t()) / temperature

            # Create labels (diagonal should be positive)
            batch_size = rec_norm.size(0)
            labels = torch.arange(batch_size, device=rec_norm.device)

            # InfoNCE loss
            loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += loss
            count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def cycle_consistency_loss(
    translations: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute cycle consistency loss with proper scaling.
    """
    total_loss = 0.0
    count = 0

    # NL -> CM -> NL cycle
    if 'nlt' in translations and 'nlt' in inputs:
        trans_norm = F.normalize(translations['nlt'], p=2, dim=1)
        inp_norm = F.normalize(inputs['nlt'], p=2, dim=1)

        cosine_sim = torch.sum(trans_norm * inp_norm, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    # CM -> NL -> CM cycle
    if 'cmt' in translations and 'cmt' in inputs:
        trans_norm = F.normalize(translations['cmt'], p=2, dim=1)
        inp_norm = F.normalize(inputs['cmt'], p=2, dim=1)

        cosine_sim = torch.sum(trans_norm * inp_norm, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def enhanced_cycle_consistency_loss(
    translations: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    margin: float = 0.1
) -> torch.Tensor:
    """
    Enhanced cycle consistency loss with margin and ranking.
    """
    total_loss = 0.0
    count = 0

    # NL -> CM -> NL cycle
    if 'nlt' in translations and 'nlt' in inputs:
        trans_norm = F.normalize(translations['nlt'], p=2, dim=1)
        inp_norm = F.normalize(inputs['nlt'], p=2, dim=1)

        # Positive similarity (should be high)
        pos_sim = torch.sum(trans_norm * inp_norm, dim=1)

        # Negative similarities (should be low)
        neg_sim = torch.mm(trans_norm, inp_norm.t())
        neg_sim = neg_sim.masked_fill(torch.eye(neg_sim.size(
            0), device=neg_sim.device).bool(), -float('inf'))
        neg_sim = torch.max(neg_sim, dim=1)[0]

        # Margin loss
        loss = torch.mean(torch.clamp(margin - pos_sim + neg_sim, min=0.0))
        total_loss += loss
        count += 1

    # CM -> NL -> CM cycle
    if 'cmt' in translations and 'cmt' in inputs:
        trans_norm = F.normalize(translations['cmt'], p=2, dim=1)
        inp_norm = F.normalize(inputs['cmt'], p=2, dim=1)

        pos_sim = torch.sum(trans_norm * inp_norm, dim=1)
        neg_sim = torch.mm(trans_norm, inp_norm.t())
        neg_sim = neg_sim.masked_fill(torch.eye(neg_sim.size(
            0), device=neg_sim.device).bool(), -float('inf'))
        neg_sim = torch.max(neg_sim, dim=1)[0]

        loss = torch.mean(torch.clamp(margin - pos_sim + neg_sim, min=0.0))
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def vector_space_preservation_loss(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute vector space preservation loss with proper scaling.
    """
    # Normalize embeddings
    source_norm = F.normalize(source_embeddings, p=2, dim=1)
    target_norm = F.normalize(target_embeddings, p=2, dim=1)

    # Compute pairwise similarities
    source_sim = torch.mm(source_norm, source_norm.t())
    target_sim = torch.mm(target_norm, target_norm.t())

    # Use cosine similarity loss instead of MSE
    cosine_sim = F.cosine_similarity(
        source_sim.flatten(), target_sim.flatten(), dim=0)
    loss = 1.0 - cosine_sim

    return loss


def enhanced_vector_space_preservation_loss(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Enhanced VSP loss with temperature scaling and ranking.
    """
    # Normalize embeddings
    source_norm = F.normalize(source_embeddings, p=2, dim=1)
    target_norm = F.normalize(target_embeddings, p=2, dim=1)

    # Compute similarity matrices
    source_sim = torch.mm(source_norm, source_norm.t()) / temperature
    target_sim = torch.mm(target_norm, target_norm.t()) / temperature

    # Apply softmax
    source_softmax = F.softmax(source_sim, dim=1)
    target_softmax = F.softmax(target_sim, dim=1)

    # KL divergence loss
    kl_loss = F.kl_div(
        target_softmax.log(),
        source_softmax,
        reduction='batchmean'
    )

    return kl_loss


def correspondence_loss(
    translations: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute correspondence loss using cosine similarity.

    This compares:
    - NL->CM translations with original CM targets
    - CM->NL translations with original NL targets
    """
    total_loss = 0.0
    count = 0

    # NL -> CM correspondence: compare NL->CM translation with original CM
    if 'cmt' in translations and 'cmt' in targets:
        trans_norm = F.normalize(translations['cmt'], p=2, dim=1)
        target_norm = F.normalize(targets['cmt'], p=2, dim=1)

        cosine_sim = torch.sum(trans_norm * target_norm, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    # CM -> NL correspondence: compare CM->NL translation with original NL
    if 'nlt' in translations and 'nlt' in targets:
        trans_norm = F.normalize(translations['nlt'], p=2, dim=1)
        target_norm = F.normalize(targets['nlt'], p=2, dim=1)

        cosine_sim = torch.sum(trans_norm * target_norm, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def ranking_loss(
    translations: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    margin: float = 0.2
) -> torch.Tensor:
    """
    Ranking loss to improve retrieval metrics.

    This compares:
    - NL->CM translations with original CM targets
    - CM->NL translations with original NL targets
    """
    total_loss = 0.0
    count = 0

    # NL -> CM ranking loss: compare NL->CM translation with original CM
    if 'cmt' in translations and 'cmt' in targets:
        trans_norm = F.normalize(translations['cmt'], p=2, dim=1)
        target_norm = F.normalize(targets['cmt'], p=2, dim=1)

        # Positive pairs (diagonal)
        pos_sim = torch.sum(trans_norm * target_norm, dim=1)

        # Negative pairs
        neg_sim = torch.mm(trans_norm, target_norm.t())
        neg_sim = neg_sim.masked_fill(torch.eye(neg_sim.size(
            0), device=neg_sim.device).bool(), -float('inf'))

        # Get hardest negative for each positive
        hardest_neg, _ = torch.max(neg_sim, dim=1)

        # Ranking loss
        loss = torch.mean(torch.clamp(margin - pos_sim + hardest_neg, min=0.0))
        total_loss += loss
        count += 1

    # CM -> NL ranking loss: compare CM->NL translation with original NL
    if 'nlt' in translations and 'nlt' in targets:
        trans_norm = F.normalize(translations['nlt'], p=2, dim=1)
        target_norm = F.normalize(targets['nlt'], p=2, dim=1)

        pos_sim = torch.sum(trans_norm * target_norm, dim=1)
        neg_sim = torch.mm(trans_norm, target_norm.t())
        neg_sim = neg_sim.masked_fill(torch.eye(neg_sim.size(
            0), device=neg_sim.device).bool(), -float('inf'))

        hardest_neg, _ = torch.max(neg_sim, dim=1)
        loss = torch.mean(torch.clamp(margin - pos_sim + hardest_neg, min=0.0))
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def adversarial_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Compute adversarial loss for generator.
    """
    if gan_type == 'least_squares':
        return torch.mean((fake_scores - 1) ** 2)
    elif gan_type == 'vanilla':
        return F.binary_cross_entropy_with_logits(
            fake_scores, torch.ones_like(fake_scores))
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def latent_adversarial_loss(
    latent_discriminator_scores: torch.Tensor,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Compute latent space adversarial loss.
    """
    if gan_type == 'least_squares':
        mean_score = torch.mean(latent_discriminator_scores)
        return torch.mean((latent_discriminator_scores - mean_score) ** 2)
    elif gan_type == 'vanilla':
        target_scores = torch.ones_like(latent_discriminator_scores) * 0.5
        return F.binary_cross_entropy_with_logits(
            latent_discriminator_scores, target_scores
        )
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Compute discriminator loss.
    """
    if gan_type == 'least_squares':
        real_loss = torch.mean((real_scores - 1) ** 2)
        fake_loss = torch.mean(fake_scores ** 2)
        return (real_loss + fake_loss) / 2
    elif gan_type == 'vanilla':
        real_loss = F.binary_cross_entropy_with_logits(
            real_scores, torch.ones_like(real_scores))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_scores, torch.zeros_like(fake_scores))
        return (real_loss + fake_loss) / 2
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def generator_loss(
    fake_scores: torch.Tensor,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Compute generator loss.
    """
    if gan_type == 'least_squares':
        return torch.mean((fake_scores - 1) ** 2)
    elif gan_type == 'vanilla':
        return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def focal_adversarial_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Focal adversarial loss to focus on hard examples.
    """
    if gan_type == 'least_squares':
        real_loss = torch.mean((real_scores - 1) ** 2)
        fake_loss = torch.mean(fake_scores ** 2)

        real_weight = alpha * (real_loss ** gamma)
        fake_weight = (1 - alpha) * (fake_loss ** gamma)

        return real_weight + fake_weight
    else:
        return torch.mean((fake_scores - 1) ** 2)


def cycle_consistency_loss_with_margin(
    translations: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    margin: float = 0.1
) -> torch.Tensor:
    """
    Cycle consistency loss with margin for better training stability.
    """
    return enhanced_cycle_consistency_loss(translations, inputs, margin)


def adaptive_loss_weighting(
    current_losses: Dict[str, torch.Tensor],
    target_losses: Dict[str, float],
) -> Dict[str, float]:
    """
    Adaptive loss weighting based on current loss magnitudes.
    """
    weights = {}

    for loss_name in current_losses:
        if loss_name in target_losses:
            current_mag = current_losses[loss_name].item()
            target_mag = target_losses[loss_name]

            ratio = target_mag / (current_mag + 1e-8)
            weights[loss_name] = max(0.1, min(10.0, ratio))

    return weights


def compute_all_losses(
    model_outputs: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    discriminator_scores: Dict[str, torch.Tensor],
    latent_discriminator_scores: torch.Tensor,
    loss_weights: Dict[str, float],
    gan_type: str = 'least_squares',
    **kwargs

) -> Dict[str, torch.Tensor]:
    """
    Compute all losses for training with improved loss functions.
    """
    reconstructions, translations = model_outputs
    enhanced = kwargs.get('enhanced', False)

    losses = {'total': torch.tensor(0.0).to(get_device())}

    # Reconstruction loss - use semantic reconstruction for better performance
    if 'reconstruction' in loss_weights:
        if enhanced:
            temperature = kwargs.get('reconstruction_temperature', 0.1)
            losses['reconstruction'] = semantic_reconstruction_loss(
                reconstructions, inputs, temperature)
        else:
            losses['reconstruction'] = reconstruction_loss(
                reconstructions, inputs)
        losses['total'] += loss_weights['reconstruction'] * \
            losses['reconstruction']

    # Cycle consistency loss
    if 'cycle_consistency' in loss_weights:
        if enhanced:
            margin = kwargs.get('margin', 0.1)
            losses['cycle_consistency'] = enhanced_cycle_consistency_loss(
                translations, inputs, margin)
        else:
            losses['cycle_consistency'] = cycle_consistency_loss(
                translations, inputs)
        losses['total'] += loss_weights['cycle_consistency'] * \
            losses['cycle_consistency']

    # Vector space preservation loss
    if 'vsp' in loss_weights:
        vsp_losses = []
        if 'nlt' in inputs and 'nlt' in translations:
            if enhanced:
                temperature = kwargs.get('temperature', 1.0)
                vsp_loss = enhanced_vector_space_preservation_loss(
                    inputs['nlt'], translations['nlt'], temperature)
            else:
                vsp_loss = vector_space_preservation_loss(
                    inputs['nlt'], translations['nlt'])
            vsp_losses.append(vsp_loss)

        if 'cmt' in inputs and 'cmt' in translations:
            if enhanced:
                temperature = kwargs.get('temperature', 1.0)
                vsp_loss = enhanced_vector_space_preservation_loss(
                    inputs['cmt'], translations['cmt'], temperature)
            else:
                vsp_loss = vector_space_preservation_loss(
                    inputs['cmt'], translations['cmt'])
            vsp_losses.append(vsp_loss)

        losses['vsp'] = torch.stack(vsp_losses).mean(
        ) if vsp_losses else torch.tensor(0.0)
        losses['total'] += loss_weights['vsp'] * losses['vsp']

    # Adversarial losses
    alpha = kwargs.get('alpha', 1.0)
    gamma = kwargs.get('gamma', 2.0)
    adversarial_fn = focal_adversarial_loss if enhanced else adversarial_loss
    adversarial_kwargs = {'alpha': alpha, 'gamma': gamma,
                          'gan_type': gan_type} if enhanced else {}

    if 'adversarial' in loss_weights:
        adv_losses = []
        fake_scores = None
        for domain, real_scores in discriminator_scores.items():
            if domain == 'nlt_output' and 'nlt' in translations:
                fake_scores = discriminator_scores.get('nlt_output_fake')
            elif domain == 'cmt_output' and 'cmt' in translations:
                fake_scores = discriminator_scores.get('cmt_output_fake')

            if fake_scores is not None:
                adv_losses.append(adversarial_fn(
                    real_scores, fake_scores, **adversarial_kwargs))

        losses['adversarial'] = torch.stack(
            adv_losses).mean() if adv_losses else torch.tensor(0.0)
        losses['total'] += loss_weights['adversarial'] * losses['adversarial']

    # Latent adversarial loss
    if 'latent_adversarial' in loss_weights:
        losses['latent_adversarial'] = latent_adversarial_loss(
            latent_discriminator_scores, gan_type=gan_type)
        losses['total'] += loss_weights['latent_adversarial'] * \
            losses['latent_adversarial']

    return losses


def compute_all_losses_with_correspondence(
    model_outputs: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    discriminator_scores: Dict[str, torch.Tensor],
    latent_discriminator_scores: torch.Tensor,
    loss_weights: Dict[str, float],
    gan_type: str = 'least_squares',
    **kwargs

) -> Dict[str, torch.Tensor]:
    """
    Compute all losses for training with correspondence and ranking losses.
    """
    reconstructions, translations = model_outputs
    enhanced = kwargs.get('enhanced', False)

    losses = {'total': torch.tensor(0.0).to(get_device())}

    # Reconstruction loss - use semantic reconstruction for better performance
    if 'reconstruction' in loss_weights:
        if enhanced:
            temperature = kwargs.get('reconstruction_temperature', 0.1)
            losses['reconstruction'] = semantic_reconstruction_loss(
                reconstructions, inputs, temperature)
        else:
            losses['reconstruction'] = reconstruction_loss(
                reconstructions, inputs)
        losses['total'] += loss_weights['reconstruction'] * \
            losses['reconstruction']

    # Cycle consistency loss
    if 'cycle_consistency' in loss_weights:
        if enhanced:
            margin = kwargs.get('margin', 0.1)
            losses['cycle_consistency'] = enhanced_cycle_consistency_loss(
                translations, inputs, margin)
        else:
            losses['cycle_consistency'] = cycle_consistency_loss(
                translations, inputs)
        losses['total'] += loss_weights['cycle_consistency'] * \
            losses['cycle_consistency']

    # Vector space preservation loss
    if 'vsp' in loss_weights:
        vsp_losses = []
        if 'nlt' in inputs and 'nlt' in translations:
            if enhanced:
                temperature = kwargs.get('temperature', 1.0)
                vsp_loss = enhanced_vector_space_preservation_loss(
                    inputs['nlt'], translations['nlt'], temperature)
            else:
                vsp_loss = vector_space_preservation_loss(
                    inputs['nlt'], translations['nlt'])
            vsp_losses.append(vsp_loss)

        if 'cmt' in inputs and 'cmt' in translations:
            if enhanced:
                temperature = kwargs.get('temperature', 1.0)
                vsp_loss = enhanced_vector_space_preservation_loss(
                    inputs['cmt'], translations['cmt'], temperature)
            else:
                vsp_loss = vector_space_preservation_loss(
                    inputs['cmt'], translations['cmt'])
            vsp_losses.append(vsp_loss)

        losses['vsp'] = torch.stack(vsp_losses).mean(
        ) if vsp_losses else torch.tensor(0.0)
        losses['total'] += loss_weights['vsp'] * losses['vsp']

    # Correspondence loss
    if 'correspondence' in loss_weights:
        losses['correspondence'] = correspondence_loss(translations, inputs)
        losses['total'] += loss_weights['correspondence'] * \
            losses['correspondence']

    # Cosine correspondence loss (alias for correspondence)
    if 'cosine_correspondence' in loss_weights:
        losses['cosine_correspondence'] = correspondence_loss(
            translations, inputs)
        losses['total'] += loss_weights['cosine_correspondence'] * \
            losses['cosine_correspondence']

    # Ranking loss
    if 'ranking' in loss_weights:
        margin = kwargs.get('ranking_margin', 0.2)
        losses['ranking'] = ranking_loss(translations, inputs, margin)
        losses['total'] += loss_weights['ranking'] * losses['ranking']

    # Adversarial losses
    alpha = kwargs.get('alpha', 1.0)
    gamma = kwargs.get('gamma', 2.0)
    adversarial_fn = focal_adversarial_loss if enhanced else adversarial_loss
    adversarial_kwargs = {'alpha': alpha, 'gamma': gamma,
                          'gan_type': gan_type} if enhanced else {}

    if 'adversarial' in loss_weights:
        adv_losses = []
        fake_scores = None
        for domain, real_scores in discriminator_scores.items():
            if domain == 'nlt_output' and 'nlt' in translations:
                fake_scores = discriminator_scores.get('nlt_output_fake')
            elif domain == 'cmt_output' and 'cmt' in translations:
                fake_scores = discriminator_scores.get('cmt_output_fake')

            if fake_scores is not None:
                adv_losses.append(adversarial_fn(
                    real_scores, fake_scores, **adversarial_kwargs))

        losses['adversarial'] = torch.stack(
            adv_losses).mean() if adv_losses else torch.tensor(0.0)
        losses['total'] += loss_weights['adversarial'] * losses['adversarial']

    # Latent adversarial loss
    if 'latent_adversarial' in loss_weights:
        losses['latent_adversarial'] = latent_adversarial_loss(
            latent_discriminator_scores, gan_type=gan_type)
        losses['total'] += loss_weights['latent_adversarial'] * \
            losses['latent_adversarial']

    return losses


# Cosine correspondence loss (kept for compatibility)
def cosine_correspondence_loss(
    translations: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute cosine correspondence loss between translations and their targets.
    """
    return correspondence_loss(translations, targets)
