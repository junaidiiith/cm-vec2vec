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
    Compute reconstruction loss.

    Args:
        reconstructions: Dictionary of reconstructed embeddings
        inputs: Dictionary of original input embeddings

    Returns:
        Reconstruction loss
    """
    total_loss = 0.0
    count = 0

    for key in reconstructions.keys():
        if key in inputs:
            loss = F.mse_loss(reconstructions[key], inputs[key])
            total_loss += loss
            count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def cycle_consistency_loss(
    translations: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute cycle consistency loss.

    Args:
        translations: Dictionary of translated embeddings
        inputs: Dictionary of original input embeddings

    Returns:
        Cycle consistency loss
    """
    total_loss = 0.0
    count = 0

    # NL -> CM -> NL cycle
    if 'nlt' in translations and 'nlt' in inputs:
        loss = F.mse_loss(translations['nlt'], inputs['nlt'])
        total_loss += loss
        count += 1

    # CM -> NL -> CM cycle
    if 'cmt' in translations and 'cmt' in inputs:
        loss = F.mse_loss(translations['cmt'], inputs['cmt'])
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def vector_space_preservation_loss(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute vector space preservation loss.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings

    Returns:
        VSP loss
    """
    # Compute pairwise similarities
    source_sim = torch.mm(source_embeddings, source_embeddings.t())
    target_sim = torch.mm(target_embeddings, target_embeddings.t())

    # Compute MSE loss between similarity matrices
    loss = F.mse_loss(source_sim, target_sim)

    return loss


def adversarial_loss(
    fake_scores: torch.Tensor,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Compute adversarial loss for generator.

    Args:
        fake_scores: Discriminator scores for fake data
        gan_type: Type of GAN loss

    Returns:
        Adversarial loss
    """
    if gan_type == 'least_squares':
        # Least squares GAN loss for generator
        # Generator wants fake scores to be close to 1 (fool discriminator)
        return torch.mean((fake_scores - 1) ** 2)
    elif gan_type == 'vanilla':
        # Vanilla GAN loss for generator
        # Generator wants fake scores to be close to 1 (fool discriminator)
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

    Args:
        latent_discriminator_scores: Latent discriminator scores
        gan_type: Type of GAN loss

    Returns:
        Latent adversarial loss
    """
    # The latent discriminator should not be able to distinguish between
    # latent representations from different domains
    # So we want the discriminator to output similar scores for all domains
    if gan_type == 'least_squares':
        # Encourage the discriminator to output similar scores for all latent reps
        # This promotes domain-agnostic latent representations
        mean_score = torch.mean(latent_discriminator_scores)
        return torch.mean((latent_discriminator_scores - mean_score) ** 2)
    elif gan_type == 'vanilla':
        # For vanilla GAN, we want the discriminator to be confused
        # about which domain the latent representations come from
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

    Args:
        real_scores: Discriminator scores for real data
        fake_scores: Discriminator scores for fake data
        gan_type: Type of GAN loss

    Returns:
        Discriminator loss
    """
    if gan_type == 'least_squares':
        # Least squares GAN loss
        # Discriminator wants real scores close to 1, fake scores close to 0
        real_loss = torch.mean((real_scores - 1) ** 2)
        fake_loss = torch.mean(fake_scores ** 2)
        return (real_loss + fake_loss) / 2
    elif gan_type == 'vanilla':
        # Vanilla GAN loss
        # Discriminator wants real scores close to 1, fake scores close to 0
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

    Args:
        fake_scores: Discriminator scores for fake data
        gan_type: Type of GAN loss

    Returns:
        Generator loss
    """
    if gan_type == 'least_squares':
        # Least squares GAN loss
        return torch.mean((fake_scores - 1) ** 2)
    elif gan_type == 'vanilla':
        # Vanilla GAN loss
        return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


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
    Compute all losses for training.

    Args:
        model_outputs: Tuple of (reconstructions, translations, latent_reps)
        inputs: Input embeddings
        discriminator_scores: Discriminator scores for each domain
        latent_discriminator_scores: Latent discriminator scores
        loss_weights: Weights for different loss components
        gan_type: Type of GAN loss
    Returns:
        Dictionary of losses
    """
    reconstructions, translations = model_outputs
    enhanced = kwargs.get('enhanced', False)

    losses = {'total': torch.tensor(0.0).to(get_device())}

    # Reconstruction loss
    if 'reconstruction' in loss_weights:
        losses['reconstruction'] = reconstruction_loss(reconstructions, inputs)
        losses['total'] += loss_weights['reconstruction'] * \
            losses['reconstruction']

    # Cycle consistency loss
    if 'cycle_consistency' in loss_weights:
        margin = kwargs.get('margin', 0.1)
        cycle_consistency_fn = cycle_consistency_loss_with_margin if enhanced else cycle_consistency_loss
        cycle_consistency_kwargs = {'margin': margin} if enhanced else {}
        losses['cycle_consistency'] = cycle_consistency_fn(
            translations, inputs, **cycle_consistency_kwargs)
        losses['total'] += loss_weights['cycle_consistency'] * \
            losses['cycle_consistency']

    # Vector space preservation loss
    vsp_kwargs = {}
    if 'vsp' in loss_weights:
        temperature = kwargs.get('temperature', 1.0)
        vsp_fn = enhanced_vector_space_preservation_loss if enhanced else vector_space_preservation_loss
        if enhanced:
            vsp_kwargs['temperature'] = temperature

        vsp_losses = []
        if 'nlt' in inputs and 'nlt' in translations:
            vsp_kwargs['source_embeddings'] = inputs['nlt']
            vsp_kwargs['target_embeddings'] = translations['nlt']
            vsp_losses.append(vsp_fn(**vsp_kwargs))

        if 'cmt' in inputs and 'cmt' in translations:
            vsp_kwargs['source_embeddings'] = inputs['cmt']
            vsp_kwargs['target_embeddings'] = translations['cmt']
            vsp_losses.append(vsp_fn(**vsp_kwargs))

        losses['vsp'] = torch.stack(vsp_losses).mean(
        ) if vsp_losses else torch.tensor(0.0)
        losses['total'] += loss_weights['vsp'] * losses['vsp']

    # Adversarial losses
    gan_type = kwargs.get('gan_type', 'least_squares')
    alpha = kwargs.get('alpha', 1.0)
    gamma = kwargs.get('gamma', 2.0)
    adversarial_fn = focal_adversarial_loss if enhanced else adversarial_loss
    adversarial_kwargs = {'alpha': alpha, 'gamma': gamma,
                          'gan_type': gan_type} if enhanced else {}
    if 'adversarial' in loss_weights:
        adv_losses = []
        fake_scores = None
        for domain, real_scores in discriminator_scores.items():
            # Get fake scores for translated embeddings
            if domain == 'nlt_output' and 'nlt' in translations:
                fake_scores = discriminator_scores.get('nlt_output_fake')
            elif domain == 'cmt_output' and 'cmt' in translations:
                fake_scores = discriminator_scores.get('cmt_output_fake')

            # Skip if no fake scores are available
            if fake_scores is not None:
                adv_losses.append(adversarial_fn(
                    real_scores, fake_scores, **adversarial_kwargs))

        # Adversarial loss
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


def enhanced_vector_space_preservation_loss(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Enhanced VSP loss with temperature scaling and stability improvements.

    Args:
        source_embeddings: Source embeddings
        target_embeddings: Target embeddings  
        temperature: Temperature for softmax scaling

    Returns:
        Enhanced VSP loss
    """
    # Normalize embeddings for stability
    source_norm = F.normalize(source_embeddings, p=2, dim=1)
    target_norm = F.normalize(target_embeddings, p=2, dim=1)

    # Compute pairwise similarities with temperature
    source_sim = torch.mm(source_norm, source_norm.t()) / temperature
    target_sim = torch.mm(target_norm, target_norm.t()) / temperature

    # Use KL divergence instead of MSE for better gradient flow
    source_softmax = F.softmax(source_sim, dim=1)
    target_softmax = F.softmax(target_sim, dim=1)

    # KL divergence loss
    kl_loss = F.kl_div(
        target_softmax.log(),
        source_softmax,
        reduction='batchmean'
    )

    return kl_loss


def focal_adversarial_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    gan_type: str = 'least_squares'
) -> torch.Tensor:
    """
    Focal adversarial loss to focus on hard examples.

    Args:
        real_scores: Real discriminator scores
        fake_scores: Fake discriminator scores
        alpha: Weighting factor
        gamma: Focusing parameter
        gan_type: Type of GAN loss

    Returns:
        Focal adversarial loss
    """
    if gan_type == 'least_squares':
        # Compute standard least squares loss
        real_loss = torch.mean((real_scores - 1) ** 2)
        fake_loss = torch.mean(fake_scores ** 2)

        # Apply focal weighting
        real_weight = alpha * (real_loss ** gamma)
        fake_weight = (1 - alpha) * (fake_loss ** gamma)

        return real_weight + fake_weight
    else:
        # Fallback to standard adversarial loss
        return torch.mean((fake_scores - 1) ** 2)


def cycle_consistency_loss_with_margin(
    translations: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    margin: float = 0.1
) -> torch.Tensor:
    """
    Cycle consistency loss with margin for better training stability.

    Args:
        translations: Translated embeddings
        inputs: Original input embeddings
        margin: Margin for loss computation

    Returns:
        Margin-based cycle consistency loss
    """
    total_loss = 0.0
    count = 0

    # NL -> CM -> NL cycle
    if 'nlt' in translations and 'nlt' in inputs:
        diff = torch.abs(translations['nlt'] - inputs['nlt'])
        # Use margin to focus on large errors
        loss = torch.mean(torch.clamp(diff - margin, min=0.0))
        total_loss += loss
        count += 1

    # CM -> NL -> CM cycle
    if 'cmt' in translations and 'cmt' in inputs:
        diff = torch.abs(translations['cmt'] - inputs['cmt'])
        loss = torch.mean(torch.clamp(diff - margin, min=0.0))
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def adaptive_loss_weighting(
    current_losses: Dict[str, torch.Tensor],
    target_losses: Dict[str, float],
) -> Dict[str, float]:
    """
    Adaptive loss weighting based on current loss magnitudes.

    Args:
        current_losses: Current loss values
        target_losses: Target loss values
        alpha: Adaptation rate

    Returns:
        Updated loss weights
    """
    weights = {}

    for loss_name in current_losses:
        if loss_name in target_losses:
            current_mag = current_losses[loss_name].item()
            target_mag = target_losses[loss_name]

            # Adjust weight based on relative magnitude
            ratio = target_mag / (current_mag + 1e-8)
            weights[loss_name] = max(0.1, min(10.0, ratio))

    return weights


# Add this new loss function to the losses.py file

def correspondence_loss(
    translations: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute correspondence loss between translations and their targets.

    This loss encourages the translated embeddings to be similar to their
    corresponding target embeddings, which is essential for semantic translation.

    Args:
        translations: Dictionary of translated embeddings
        targets: Dictionary of target embeddings

    Returns:
        Correspondence loss
    """
    total_loss = 0.0
    count = 0

    # NL -> CM correspondence
    if 'cmt' in translations and 'cmt' in targets:
        loss = F.mse_loss(translations['cmt'], targets['cmt'])
        total_loss += loss
        count += 1

    # CM -> NL correspondence
    if 'nlt' in translations and 'nlt' in targets:
        loss = F.mse_loss(translations['nlt'], targets['nlt'])
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


def cosine_correspondence_loss(
    translations: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute cosine correspondence loss between translations and their targets.

    This loss encourages high cosine similarity between translated embeddings
    and their corresponding target embeddings.

    Args:
        translations: Dictionary of translated embeddings
        targets: Dictionary of target embeddings

    Returns:
        Cosine correspondence loss (1 - cosine_similarity)
    """
    total_loss = 0.0
    count = 0

    # NL -> CM correspondence
    if 'cmt' in translations and 'cmt' in targets:
        # Normalize embeddings
        trans_norm = F.normalize(translations['cmt'], p=2, dim=1)
        target_norm = F.normalize(targets['cmt'], p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = torch.sum(trans_norm * target_norm, dim=1)

        # Loss is 1 - cosine_similarity (we want high similarity)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    # CM -> NL correspondence
    if 'nlt' in translations and 'nlt' in targets:
        # Normalize embeddings
        trans_norm = F.normalize(translations['nlt'], p=2, dim=1)
        target_norm = F.normalize(targets['nlt'], p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = torch.sum(trans_norm * target_norm, dim=1)

        # Loss is 1 - cosine_similarity (we want high similarity)
        loss = torch.mean(1.0 - cosine_sim)
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0)


# Update the compute_all_losses function to include correspondence loss

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
    Compute all losses for training.

    Args:
        model_outputs: Tuple of (reconstructions, translations, latent_reps)
        inputs: Input embeddings
        discriminator_scores: Discriminator scores for each domain
        latent_discriminator_scores: Latent discriminator scores
        loss_weights: Weights for different loss components
        gan_type: Type of GAN loss
    Returns:
        Dictionary of losses
    """

    _, translations = model_outputs
    losses = compute_all_losses(model_outputs, inputs, discriminator_scores,
                                latent_discriminator_scores, loss_weights, gan_type, **kwargs)

    # Correspondence loss
    # Correspondence loss - NEW: This is crucial for semantic translation
    if 'correspondence' in loss_weights:
        losses['correspondence'] = correspondence_loss(translations, inputs)
        losses['total'] += loss_weights['correspondence'] * \
            losses['correspondence']

    # Cosine correspondence loss - Alternative to MSE correspondence
    if 'cosine_correspondence' in loss_weights:
        losses['cosine_correspondence'] = cosine_correspondence_loss(
            translations, inputs)
        losses['total'] += loss_weights['cosine_correspondence'] * \
            losses['cosine_correspondence']

    return losses
