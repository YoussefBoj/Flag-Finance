import torch

def print_state_dict_shapes(state_dict):
    for k, v in state_dict.items():
        print(f"{k}: {tuple(v.shape)}")

def debug_checkpoint(checkpoint_path, ModelClass, model_name):
    print(f"\n--- Debugging {model_name} ---")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
    else:
        checkpoint_state_dict = checkpoint
    
    # Initialize model with assumed dims (adjust if you know different)
    model = ModelClass(gnn_dim=384, lstm_dim=256, hidden_dim=256)
    
    print(f"\nCheckpoint keys ({len(checkpoint_state_dict)}):")
    print(list(checkpoint_state_dict.keys()))
    
    model_state_dict = model.state_dict()
    print(f"\nModel keys ({len(model_state_dict)}):")
    print(list(model_state_dict.keys()))
    
    missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
    unexpected_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())
    
    print(f"\nMissing keys in checkpoint:")
    for k in missing_keys:
        print(f"  {k}")
    
    print(f"\nUnexpected keys in checkpoint:")
    for k in unexpected_keys:
        print(f"  {k}")
    
    # Try to load with strict=False to see if loads partially
    try:
        model.load_state_dict(checkpoint_state_dict, strict=False)
        print("\nCheckpoint loaded with strict=False successfully")
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")

if __name__ == "__main__":
    # Import your fusion models here
    from src.models.fusion import CrossModalFusion, SimpleFusion, GatedFusion, AttentionFusion

    checkpoint_path = "data/models/CrossModalFusion_best.pt"  # Change as necessary# Change as necessary
    checkpoint_path1 = "data/models/SimpleFusion_best.pt"  # Change as necessary
    checkpoint_path2 = "data/models/GatedFusion_best.pt"  # Change as necessary
    checkpoint_path3 = "data/models/AttentionFusion_best.pt"  # Change as necessary

    debug_checkpoint(checkpoint_path, CrossModalFusion, "CrossModalFusion")
    debug_checkpoint(checkpoint_path1, SimpleFusion, "SimpleFusion")
    debug_checkpoint(checkpoint_path2, GatedFusion, "GatedFusion")
    debug_checkpoint(checkpoint_path3, AttentionFusion, "AttentionFusion")
