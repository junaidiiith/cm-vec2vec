"""
NL2CM training example for CMVec2Vec
"""

import os
from pathlib import Path

from cm_vec2vec import (
    CMVec2VecTranslator, 
    CMVec2VecTrainer, 
    CMVec2VecEvaluator
)
from cm_vec2vec.data_loader import load_nl2cm_data
from cm_vec2vec.parse_args import parse_args


def main():
    args = parse_args()

    print("CMVec2Vec Training Configuration:")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Adapter depth: {args.adapter_depth}")
    print(f"  Backbone depth: {args.backbone_depth}")
    print(f"  Dropout rate: {args.dropout}")
    print(f"  Use conditioning: {args.use_conditioning}")
    print(f"  Normalize embeddings: {args.normalize_embeddings}")
    print(f"  Weight initialization: {args.weight_init}")
    print(f"  Activation function: {args.activation}")
    print()


    # Load data
    data_path = os.path.join(args.data_path, args.dataset_name)
    nl_cm_cols = [args.nl_col, args.cm_col]
    print("Loading CMVec2Vec data...")
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path=data_path,
        nl_cm_cols=nl_cm_cols,
        test_size=args.test_size,
        random_state=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()

    # Create model
    print("Creating CMVec2Vec model...")
    model = CMVec2VecTranslator(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        adapter_depth=args.adapter_depth,
        backbone_depth=args.backbone_depth,
        dropout=args.dropout,
        use_conditioning=args.use_conditioning,
        normalize_embeddings=args.normalize_embeddings,
        weight_init=args.weight_init,
        activation=args.activation
    )

    print(
        f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create trainer
    print("Creating trainer...")
    trainer = CMVec2VecTrainer(
        model=model,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        loss_weights=args.loss_weights,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps
    )

    print("Starting CMVec2Vec training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience
    )

    print("Training completed!")
    print(f"Final training loss: {history['train_losses'][-1]['total']:.4f}")
    if history['val_losses']:
        print(
            f"Final validation loss: {history['val_losses'][-1]['total']:.4f}")

    # Evaluation
    print("\nEvaluating model...")
    evaluator = CMVec2VecEvaluator(model)

    # Evaluate CMVec2Vec translation
    results = evaluator.evaluate_loader(test_loader)

    print("\nEvaluation Results:")
    print(evaluator.create_evaluation_table(results))

    # Save evaluation results
    eval_save_path = Path(args.save_dir) / 'evaluation_results.json'
    eval_save_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(eval_save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to: {eval_save_path}")


if __name__ == '__main__':
    main()
