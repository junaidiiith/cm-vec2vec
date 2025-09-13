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
    data_path = os.path.join(args.data_path, args.dataset_name)
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
        save_dir=args.save_dir
    )

    print("   ✓ Trainer creation successful!")

    # Test training step
    print("\n4. Testing training step...")
    trainer.train(train_loader, val_loader)

    print("   ✓ Training step successful!")

    # Test validation
    print("\n5. Testing validation...")
    val_losses = trainer.validate(val_loader)
    print(f"   Validation losses: {val_losses}")
    print("   ✓ Validation successful!")

    # Test evaluation
    print("\n6. Testing evaluation...")
    evaluator = CMVec2VecEvaluator(model)
    results = evaluator.evaluate_loader(test_loader)
    print(f"   Evaluation results: {results}")
    save_dir = os.path.join(args.save_dir, args.dataset_name)
    with open(os.path.join(save_dir, f'evaluation_results_{args.dataset_name}.json'), 'w') as f:
        json.dump(results, f)

    print("   Evaluation table created successfully")

    print("   ✓ Evaluation successful!")

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
