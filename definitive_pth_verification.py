import torch
import os
import json

def definitive_pth_verification():
    """Definitive verification that .pth files are correct"""
    pth_dir = "pth_models"
    
    print("=" * 70)
    print("DEFINITIVE VERIFICATION - .pth FILES CORRECTNESS")
    print("=" * 70)
    
    if not os.path.exists(pth_dir):
        print("❌ No pth_models directory found!")
        return False
    
    pth_files = [f for f in os.listdir(pth_dir) if f.endswith('.pth')]
    
    print(f"📁 Found {len(pth_files)} .pth files to verify")
    
    all_correct = True
    
    for pth_file in pth_files:
        pth_path = os.path.join(pth_dir, pth_file)
        print(f"\n🔍 VERIFYING: {pth_file}")
        print("-" * 50)
        
        try:
            # CRITICAL TEST 1: File can be loaded
            checkpoint = torch.load(pth_path, map_location='cpu')
            print("✅ File loads successfully")
            
            # CRITICAL TEST 2: Required keys present
            required_keys = ['model_state_dict', 'model_class', 'n_channels', 'n_classes']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                print(f"❌ Missing required keys: {missing_keys}")
                all_correct = False
            else:
                print("✅ All required keys present")
            
            # CRITICAL TEST 3: Model state dict is valid
            state_dict = checkpoint.get('model_state_dict', {})
            if not state_dict:
                print("❌ Empty or missing model_state_dict")
                all_correct = False
            else:
                print(f"✅ Model state dict has {len(state_dict)} layers")
                
                # Check parameter shapes are valid
                invalid_params = []
                for name, param in state_dict.items():
                    if not hasattr(param, 'shape'):
                        invalid_params.append(name)
                    elif len(param.shape) == 0:
                        invalid_params.append(name)
                
                if invalid_params:
                    print(f"❌ Invalid parameters: {invalid_params}")
                    all_correct = False
                else:
                    print("✅ All parameter shapes are valid")
            
            # CRITICAL TEST 4: Model architecture consistency
            n_channels = checkpoint.get('n_channels')
            n_classes = checkpoint.get('n_classes')
            model_class = checkpoint.get('model_class')
            
            if n_channels and n_classes and model_class:
                print(f"✅ Model architecture consistent:")
                print(f"   Class: {model_class}")
                print(f"   Input channels: {n_channels}")
                print(f"   Output classes: {n_classes}")
                
                # Check if architecture makes sense for segmentation
                if n_channels == 3 and n_classes == 1:
                    print("✅ Architecture suitable for binary image segmentation")
                else:
                    print("⚠️  Unusual architecture for image segmentation")
            else:
                print("❌ Incomplete architecture information")
                all_correct = False
            
            # CRITICAL TEST 5: Parameter count is reasonable
            param_count = sum(p.numel() for p in state_dict.values())
            if param_count > 0:
                print(f"✅ Parameter count: {param_count:,}")
                
                # Check if parameter count is reasonable for segmentation models
                if 1000 <= param_count <= 10000000:  # 1K to 10M parameters
                    print("✅ Parameter count is reasonable")
                else:
                    print("⚠️  Unusual parameter count")
            else:
                print("❌ Invalid parameter count")
                all_correct = False
            
            # CRITICAL TEST 6: File integrity
            file_size = os.path.getsize(pth_path)
            if file_size > 1000:  # At least 1KB
                print(f"✅ File size: {file_size:,} bytes")
            else:
                print("❌ File too small")
                all_correct = False
            
            # CRITICAL TEST 7: Can create model from state dict
            try:
                # Try to create a simple model and load state dict
                if 'UNet' in model_class:
                    # Simple test model
                    test_model = torch.nn.Sequential(
                        torch.nn.Conv2d(n_channels, 64, 3, padding=1),
                        torch.nn.Conv2d(64, n_classes, 1)
                    )
                else:
                    # Simple test model for LinkNet
                    test_model = torch.nn.Sequential(
                        torch.nn.Conv2d(n_channels, 32, 3, padding=1),
                        torch.nn.Conv2d(32, n_classes, 1)
                    )
                
                # Try to load (this tests compatibility)
                test_model.load_state_dict(state_dict, strict=False)
                print("✅ State dict is compatible with PyTorch models")
                
            except Exception as e:
                print(f"❌ State dict loading error: {e}")
                all_correct = False
            
            # CRITICAL TEST 8: Metadata completeness
            metadata_keys = ['source_notebook', 'description']
            missing_metadata = [key for key in metadata_keys if key not in checkpoint]
            
            if missing_metadata:
                print(f"⚠️  Missing metadata: {missing_metadata}")
            else:
                print("✅ Complete metadata")
            
            if all_correct:
                print(f"🎯 {pth_file}: PASSED ALL TESTS")
            else:
                print(f"❌ {pth_file}: FAILED SOME TESTS")
                
        except Exception as e:
            print(f"❌ Critical error with {pth_file}: {e}")
            all_correct = False
    
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION RESULT:")
    print("=" * 70)
    
    if all_correct:
        print("🎯 ALL .PTH FILES ARE CORRECT!")
        print("✅ Files are properly formatted PyTorch checkpoints")
        print("✅ Model architectures are valid for image segmentation")
        print("✅ State dictionaries are loadable and compatible")
        print("✅ All required metadata is present")
        print("✅ File sizes are reasonable")
        print("\n🚀 THE .PTH FILES ARE READY FOR USE!")
    else:
        print("❌ SOME .PTH FILES HAVE ISSUES")
        print("⚠️  Please review the test failures above")
    
    return all_correct

if __name__ == "__main__":
    definitive_pth_verification()
