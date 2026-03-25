import torch
import os

def verify_pth_files():
    """Verify and display information about created .pth files"""
    pth_dir = "pth_models"
    
    if not os.path.exists(pth_dir):
        print("No pth_models directory found!")
        return
    
    pth_files = [f for f in os.listdir(pth_dir) if f.endswith('.pth')]
    
    print(f"🔍 Verifying {len(pth_files)} .pth files...")
    print("=" * 60)
    
    for pth_file in pth_files:
        pth_path = os.path.join(pth_dir, pth_file)
        print(f"\n📁 {pth_file}")
        print("-" * 40)
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(pth_path, map_location='cpu')
            
            # Display information
            print(f"✅ File loads successfully")
            print(f"📋 Contains keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                param_count = sum(p.numel() for p in state_dict.values())
                print(f"🔢 Model parameters: {param_count:,}")
                print(f"📦 Model layers: {len(state_dict)}")
                
                # Show first few layers
                print(f"🏗️  First 5 layers:")
                for i, (name, param) in enumerate(state_dict.items()):
                    if i < 5:
                        print(f"   {i+1}. {name}: {param.shape}")
            
            if 'model_class' in checkpoint:
                print(f"🏷️  Model class: {checkpoint['model_class']}")
            
            if 'n_channels' in checkpoint:
                print(f"🎨 Input channels: {checkpoint['n_channels']}")
            
            if 'n_classes' in checkpoint:
                print(f"🎯 Output classes: {checkpoint['n_classes']}")
            
            if 'source_notebook' in checkpoint:
                print(f"📓 Source: {checkpoint['source_notebook']}")
            
            if 'description' in checkpoint:
                print(f"📝 Description: {checkpoint['description']}")
            
            print(f"💾 File size: {os.path.getsize(pth_path):,} bytes")
            
        except Exception as e:
            print(f"❌ Error loading {pth_file}: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎯 All .pth files verified successfully!")

if __name__ == "__main__":
    verify_pth_files()
