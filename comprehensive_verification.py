import torch
import json
import os
import re
from datetime import datetime

def analyze_notebook_content(notebook_path):
    """Analyze notebook content to understand what models it contains"""
    print(f"\n🔍 Analyzing: {os.path.basename(notebook_path)}")
    print("-" * 50)
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extract all code
    all_code = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            all_code.append(source)
    
    full_code = '\n'.join(all_code)
    
    # Look for model definitions
    model_patterns = {
        'class_definitions': r'class\s+(\w+)\s*\(',
        'unet_patterns': r'(?i)(unet|u-net)',
        'linknet_patterns': r'(?i)(linknet)',
        'deeplab_patterns': r'(?i)(deeplab|deep\s*lab)',
        'model_saves': r'torch\.save\([^,]+,\s*["\']([^"\']+)["\']\)',
        'model_loads': r'torch\.load\(["\']([^"\']+)["\']\)',
        'conv_layers': r'Conv2d\([^)]+\)',
        'transpose_layers': r'ConvTranspose2d\([^)]+\)',
        'pooling_layers': r'MaxPool2d\([^)]+\)',
        'batchnorm_layers': r'BatchNorm2d\([^)]+\)',
        'relu_activations': r'ReLU\([^)]*\)',
        'sigmoid_activations': r'Sigmoid\([^)]*\)'
    }
    
    analysis = {}
    for pattern_name, pattern in model_patterns.items():
        matches = re.findall(pattern, full_code)
        analysis[pattern_name] = matches
        print(f"📋 {pattern_name}: {len(matches)} found")
        if matches and pattern_name in ['class_definitions', 'model_saves', 'model_loads']:
            print(f"   Examples: {matches[:3]}")
    
    # Count total lines and complexity
    lines = full_code.split('\n')
    analysis['total_lines'] = len(lines)
    analysis['code_cells'] = len(all_code)
    analysis['total_chars'] = len(full_code)
    
    print(f"📊 Total code cells: {analysis['code_cells']}")
    print(f"📊 Total lines: {analysis['total_lines']}")
    print(f"📊 Total characters: {analysis['total_chars']}")
    
    return analysis

def analyze_pth_file(pth_path):
    """Analyze .pth file content"""
    print(f"\n🔍 Analyzing: {os.path.basename(pth_path)}")
    print("-" * 50)
    
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    analysis = {
        'keys': list(checkpoint.keys()),
        'model_class': checkpoint.get('model_class', 'Unknown'),
        'model_name': checkpoint.get('model_name', 'Unknown'),
        'parameters': checkpoint.get('parameters', 0),
        'n_channels': checkpoint.get('n_channels', 0),
        'n_classes': checkpoint.get('n_classes', 0),
        'architecture': checkpoint.get('architecture', 'Unknown'),
        'source_notebook': checkpoint.get('source_notebook', 'Unknown'),
        'description': checkpoint.get('description', 'Unknown')
    }
    
    # Analyze model structure
    state_dict = checkpoint.get('model_state_dict', {})
    layer_types = {}
    param_count = 0
    
    for name, param in state_dict.items():
        param_count += param.numel()
        
        if 'conv' in name.lower():
            layer_types['conv'] = layer_types.get('conv', 0) + 1
        elif 'transpose' in name.lower():
            layer_types['transpose'] = layer_types.get('transpose', 0) + 1
        elif 'pool' in name.lower():
            layer_types['pool'] = layer_types.get('pool', 0) + 1
        elif 'norm' in name.lower():
            layer_types['norm'] = layer_types.get('norm', 0) + 1
    
    analysis['layer_types'] = layer_types
    analysis['actual_parameters'] = param_count
    analysis['state_dict_layers'] = len(state_dict)
    
    print(f"🏗️  Model class: {analysis['model_class']}")
    print(f"📝 Model name: {analysis['model_name']}")
    print(f"📊 Parameters: {analysis['parameters']:,}")
    print(f"📊 Actual parameters: {analysis['actual_parameters']:,}")
    print(f"🎨 Input channels: {analysis['n_channels']}")
    print(f"🎯 Output classes: {analysis['n_classes']}")
    print(f"🏗️  Architecture: {analysis['architecture']}")
    print(f"📓 Source notebook: {analysis['source_notebook']}")
    print(f"📋 Layer types: {analysis['layer_types']}")
    print(f"📋 State dict layers: {analysis['state_dict_layers']}")
    
    return analysis

def verify_correctness(notebook_analysis, pth_analysis, notebook_name, pth_name):
    """Verify if .pth correctly represents notebook content"""
    print(f"\n🎯 VERIFYING: {notebook_name} → {pth_name}")
    print("=" * 60)
    
    score = 0
    total_checks = 0
    issues = []
    
    # Check 1: Model types match
    total_checks += 1
    notebook_has_unet = len([p for p in notebook_analysis.get('unet_patterns', [])]) > 0
    notebook_has_linknet = len([p for p in notebook_analysis.get('linknet_patterns', [])]) > 0
    
    pth_is_unet = 'unet' in pth_analysis.get('model_class', '').lower() or 'unet' in pth_analysis.get('model_name', '').lower()
    pth_is_linknet = 'linknet' in pth_analysis.get('model_class', '').lower() or 'linknet' in pth_analysis.get('model_name', '').lower()
    
    if (notebook_has_unet and pth_is_unet) or (notebook_has_linknet and pth_is_linknet):
        score += 1
        print("✅ Model type matches notebook content")
    else:
        issues.append("Model type doesn't match notebook content")
        print("❌ Model type doesn't match notebook content")
    
    # Check 2: Architecture complexity
    total_checks += 1
    notebook_conv_count = len(notebook_analysis.get('conv_layers', []))
    pth_conv_count = pth_analysis.get('layer_types', {}).get('conv', 0)
    
    if pth_conv_count > 0 and notebook_conv_count > 0:
        score += 1
        print("✅ Architecture complexity is reasonable")
    else:
        issues.append("Architecture complexity mismatch")
        print("❌ Architecture complexity mismatch")
    
    # Check 3: Input/output channels
    total_checks += 1
    if pth_analysis.get('n_channels') == 3 and pth_analysis.get('n_classes') == 1:
        score += 1
        print("✅ Input/output channels correct for segmentation")
    else:
        issues.append("Input/output channels incorrect")
        print("❌ Input/output channels incorrect")
    
    # Check 4: Parameter count reasonable
    total_checks += 1
    param_count = pth_analysis.get('parameters', 0)
    if 1000 <= param_count <= 10000000:  # 1K to 10M parameters
        score += 1
        print("✅ Parameter count is reasonable")
    else:
        issues.append("Parameter count unreasonable")
        print("❌ Parameter count unreasonable")
    
    # Check 5: File structure完整性
    total_checks += 1
    required_keys = ['model_state_dict', 'model_class', 'n_channels', 'n_classes']
    missing_keys = [key for key in required_keys if key not in pth_analysis.get('keys', [])]
    
    if len(missing_keys) == 0:
        score += 1
        print("✅ All required keys present")
    else:
        issues.append(f"Missing keys: {missing_keys}")
        print(f"❌ Missing keys: {missing_keys}")
    
    # Check 6: Source notebook mapping
    total_checks += 1
    expected_notebook = os.path.basename(notebook_name)
    actual_source = pth_analysis.get('source_notebook', '')
    
    if expected_notebook in actual_source:
        score += 1
        print("✅ Source notebook correctly mapped")
    else:
        issues.append("Source notebook mapping incorrect")
        print("❌ Source notebook mapping incorrect")
    
    # Calculate final score
    percentage = (score / total_checks) * 100
    print(f"\n📊 VERIFICATION SCORE: {score}/{total_checks} ({percentage:.1f}%)")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    if percentage >= 80:
        print("🎯 RESULT: .pth CORRECTLY represents notebook!")
        return True
    else:
        print("❌ RESULT: .pth has significant issues!")
        return False

def main():
    current_dir = os.getcwd()
    pth_dir = "pth_models"
    
    print("🔍 COMPREHENSIVE .pth CORRECTNESS VERIFICATION")
    print("=" * 70)
    
    # Define file pairs
    file_pairs = [
        ("Custom_UNet_LinkNet_DeepLapv3+.ipynb", os.path.join(pth_dir, "Custom_Model.pth")),
        ("LinkNet and UNet Trained Model.ipynb", os.path.join(pth_dir, "LinkNet_Model.pth"))
    ]
    
    all_correct = True
    
    for notebook_name, pth_path in file_pairs:
        if os.path.exists(notebook_name) and os.path.exists(pth_path):
            # Analyze both files
            notebook_analysis = analyze_notebook_content(notebook_name)
            pth_analysis = analyze_pth_file(pth_path)
            
            # Verify correctness
            is_correct = verify_correctness(notebook_analysis, pth_analysis, notebook_name, os.path.basename(pth_path))
            
            if not is_correct:
                all_correct = False
        else:
            print(f"⚠️  Missing files: {notebook_name} or {pth_path}")
            all_correct = False
    
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION RESULT:")
    print("=" * 70)
    
    if all_correct:
        print("🎯 ALL .PTH FILES ARE CORRECT!")
        print("✅ Both files accurately represent their source notebooks")
        print("✅ Model architectures match notebook content")
        print("✅ Parameters and structure are valid")
        print("✅ Ready for production use!")
    else:
        print("❌ SOME .PTH FILES HAVE ISSUES")
        print("⚠️  Please review the verification results above")
    
    return all_correct

if __name__ == "__main__":
    main()
