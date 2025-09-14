"""
End-to-end test for CMVec2Vec with actual data loading and training
"""

import json
import os
from cm_vec2vec.data_loader import load_nl2cm_data
from cm_vec2vec import CMVec2VecTranslator, CMVec2VecTrainer, CMVec2VecEvaluator
from cm_vec2vec.parse_args import parse_args


def test_end_to_end():
    """Test complete end-to-end workflow."""

    args = parse_args()
    print("CMVec2Vec End-to-End Test")
    print("=" * 50)
    data_path = os.path.join(args.data_path, args.dataset)
    nl_cm_cols = [args.nl_col, args.cm_col]

    # Test data loading
    print("1. Testing data loading...")
    train_loader, val_loader, test_loader = load_nl2cm_data(
        data_path=data_path,
        nl_cm_cols=nl_cm_cols,
        test_size=args.test_size,
        random_state=args.seed,
        num_workers=args.num_workers
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test a few batches
    for i, batch in enumerate(train_loader):
        print(
            f"   Batch {i+1} shapes: {[(k, v.shape) for k, v in batch.items()]}")
        if i >= 2:  # Test only first few batches
            break

    print("   ✓ Data loading successful!")

    # Test model creation
    print("\n2. Testing model creation...")
    model = CMVec2VecTranslator(
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        adapter_depth=args.adapter_depth,
        backbone_depth=args.backbone_depth
    )

    print(
        f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   ✓ Model creation successful!")

    # Test trainer creation
    print("\n3. Testing trainer creation...")
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    evaluator = CMVec2VecEvaluator(model=model, save_dir=save_dir)
    trainer = CMVec2VecTrainer(
        model=model,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        loss_weights={
            'reconstruction': args.reconstruction_weight,
            'cycle_consistency': args.cycle_consistency_weight,
            'vsp': args.vsp_weight,
            'adversarial': args.adversarial_weight,
            'latent_adversarial': args.latent_adversarial_weight
        },
        save_dir=save_dir,
        evaluator=evaluator
    )
    print("   ✓ Trainer and evaluator creation successful!")
    print("Loss Weights:")
    print(f"   Reconstruction: {args.reconstruction_weight}")
    print(f"   Cycle Consistency: {args.cycle_consistency_weight}")
    print(f"   VSP: {args.vsp_weight}")
    print(f"   Adversarial: {args.adversarial_weight}")
    print(f"   Latent Adversarial: {args.latent_adversarial_weight}")
    print(f"   Enhanced Losses: {args.enhanced_losses}")
    print(f"   Save Table: {args.save_table}")
    print(f"   Save Plots: {args.save_plots}")
    print(f"   Save Dir: {save_dir}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Data Path: {data_path}")
    print(f"   NL Col: {args.nl_col}")
    print(f"   CM Col: {args.cm_col}")
    print(f"   Test Size: {args.test_size}")
    print(f"   Seed: {args.seed}")
    print(f"   Num Workers: {args.num_workers}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Save Every: {args.save_every}")
    print(f"   Early Stopping Patience: {args.early_stopping_patience}")
    print(f"   LR Generator: {args.lr_generator}")
    print(f"   LR Discriminator: {args.lr_discriminator}")
    print(f"   Embedding Dim: {args.embedding_dim}")
    print(f"   Latent Dim: {args.latent_dim}")
    print(f"   Hidden Dim: {args.hidden_dim}")
    print(f"   Adapter Depth: {args.adapter_depth}")
    print(f"   Backbone Depth: {args.backbone_depth}")
    print(f"   Save Dir: {save_dir}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Data Path: {data_path}")
    print(f"   NL Col: {args.nl_col}")
    print(f"   CM Col: {args.cm_col}")
    print(f"   Test Size: {args.test_size}")
    print(f"   Seed: {args.seed}")
    print(f"   Num Workers: {args.num_workers}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Save Every: {args.save_every}")
    print(f"   Early Stopping Patience: {args.early_stopping_patience}")
    print(f"   LR Generator: {args.lr_generator}")
    print(f"   LR Discriminator: {args.lr_discriminator}")

    # Test training step
    print("\n4. Testing training step...")
    train_fn = trainer.enhanced_train if args.enhanced_losses else trainer.train
    train_fn(
        train_loader, val_loader, epochs=args.epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience
    )

    print("   ✓ Training successful!")

    # Test validation
    print("\n5. Testing validation...")
    val_fn = trainer.enhanced_validate if args.enhanced_losses else trainer.validate
    val_losses = val_fn(val_loader, save_table=args.save_table)
    print(f"   Validation losses: {val_losses}")
    print(f"   ✓ Validation successful! with losses: {val_losses}")

    # Test evaluation
    print("\n6. Testing evaluation...")

    results = evaluator.evaluate_loader(test_loader, plot=args.save_plots, save_table=args.save_table)
    print(f"   Evaluation results: {results}")
    with open(os.path.join(save_dir, f'evaluation_results.json'), 'w') as f:
        json.dump(results, f)

    print("   Evaluation table created successfully")

    print(f"   ✓ Evaluation successful! with results: {results}")

    # Test translation

    print("\n" + "=" * 50)
    print("✓ All end-to-end tests passed successfully!")
    print("CMVec2Vec is ready for production use!")

    return True


if __name__ == "__main__":
    try:
        test_end_to_end()
    except Exception as e:
        print(f"\n❌ End-to-end test failed with error: {e}")
        import traceback
        import sys
        traceback.print_exc()
        sys.exit(1)
