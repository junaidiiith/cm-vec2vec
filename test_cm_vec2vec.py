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
        num_workers=args.num_workers,
        batch_size=args.batch_size
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
    model_kwargs = {
        'embedding_dim': args.embedding_dim,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'adapter_depth': args.adapter_depth,
        'backbone_depth': args.backbone_depth
    }
    model = CMVec2VecTranslator(**model_kwargs)

    print(
        f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}",
        f"   Model kwargs: {model_kwargs}"
    )
    print("   ✓ Model creation successful!")

    # Test trainer creation
    print("\n3. Testing trainer creation...")
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    evaluator = CMVec2VecEvaluator(model=model, save_dir=save_dir)
    enhanced_losses_kwargs = {
        'vsp_temperature': args.vsp_temperature,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'cycle_margin': args.cycle_margin,
        'enhanced': args.enhanced_losses
    }
    trainer = CMVec2VecTrainer(
        model=model,
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        loss_weights={
            'reconstruction': args.reconstruction_weight,
            'cycle_consistency': args.cycle_consistency_weight,
            'vsp': args.vsp_weight,
            'adversarial': args.adversarial_weight,
            'latent_adversarial': args.latent_adversarial_weight,
            'correspondence': args.correspondence_weight,
            'cosine_correspondence': args.cosine_correspondence_weight,
            'ranking': args.ranking_weight
        },
        save_dir=save_dir,
        evaluator=evaluator,
        **enhanced_losses_kwargs
    )
    print("   ✓ Trainer and evaluator creation successful!")
    # Test training step
    print("\n4. Testing training step...")
    trainer.train(
        train_loader, val_loader, epochs=args.epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience,
        save_table=args.save_table
    )

    print("   ✓ Training successful!")

    # Test validation
    print("\n5. Testing validation...")
    val_losses = trainer.validate(val_loader, save_table=args.save_table, compute_metrics=True)
    print(f"   Validation losses: {val_losses}")
    print(f"   ✓ Validation successful! with losses: {val_losses}")

    # Test evaluation
    print("\n6. Testing evaluation...")
    results = evaluator.evaluate_loader(test_loader, plot=True, save_table=True)
    print(f"   Evaluation results: {results}")
    with open(os.path.join(save_dir, f'evaluation_results.json'), 'w') as f:
        results = {k: float(v) for k, v in results.items()}
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
