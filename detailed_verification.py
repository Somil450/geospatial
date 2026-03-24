import h5py
import json
import os

def detailed_verification(ipynb_path, h5_path):
    """Detailed verification of notebook to H5 conversion"""
    print(f"\n=== Detailed Verification: {os.path.basename(ipynb_path)} ===")
    
    # Load original notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get all code cells from original
    original_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            original_cells.append(source)
    
    print(f"Original notebook has {len(original_cells)} code cells")
    
    # Load H5 file
    with h5py.File(h5_path, 'r') as h5file:
        h5_cells = []
        code_group = h5file['code_cells']
        
        # Get cells in order
        cell_keys = sorted([key for key in code_group.keys() if key.startswith('cell_')])
        
        for cell_key in cell_keys:
            cell = code_group[cell_key]
            
            # Reconstruct source
            source_lines = []
            source_keys = sorted([key for key in cell.keys() if key.startswith('source_line_')])
            
            for source_key in source_keys:
                line_data = cell[source_key][()]
                if isinstance(line_data, bytes):
                    source_lines.append(line_data.decode('utf-8'))
                else:
                    source_lines.append(str(line_data))
            
            h5_cells.append('\n'.join(source_lines))
    
    print(f"H5 file has {len(h5_cells)} code cells")
    
    # Compare each cell
    matches = 0
    differences = []
    
    for i, (orig, h5) in enumerate(zip(original_cells, h5_cells)):
        if orig.strip() == h5.strip():
            matches += 1
        else:
            differences.append(i)
            print(f"\n❌ Cell {i} differs:")
            print(f"Original length: {len(orig)} chars")
            print(f"H5 length: {len(h5)} chars")
            print(f"Original first 100 chars: {repr(orig[:100])}")
            print(f"H5 first 100 chars: {repr(h5[:100])}")
    
    print(f"\n=== Summary ===")
    print(f"Total cells: {len(original_cells)}")
    print(f"Matching cells: {matches}")
    print(f"Different cells: {len(differences)}")
    print(f"Match rate: {matches/len(original_cells)*100:.1f}%")
    
    if len(differences) > 0:
        print(f"\nCell indices with differences: {differences[:10]}{'...' if len(differences) > 10 else ''}")
    
    return matches == len(original_cells)

def check_h5_structure(h5_path):
    """Check H5 file structure in detail"""
    print(f"\n=== H5 Structure: {os.path.basename(h5_path)} ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        print("Root attributes:")
        for attr_name, attr_value in h5file.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        print("\nGroups:")
        def print_groups(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
                for attr_name, attr_value in obj.attrs.items():
                    print(f"    Attr: {attr_name} = {attr_value}")
                
                # Count datasets
                datasets = [key for key in obj.keys() if isinstance(obj[key], h5py.Dataset)]
                if datasets:
                    print(f"    Datasets: {len(datasets)}")
                    if len(datasets) <= 3:  # Show first few
                        for ds_name in datasets[:3]:
                            ds = obj[ds_name]
                            data = ds[()]
                            if isinstance(data, bytes):
                                content = data.decode('utf-8')[:50]
                            else:
                                content = str(data)[:50]
                            print(f"      {ds_name}: {repr(content)}")
        
        h5file.visititems(print_groups)

def main():
    current_dir = os.getcwd()
    
    # Find pairs of .ipynb and .h5 files
    ipynb_files = [f for f in os.listdir(current_dir) if f.endswith('.ipynb')]
    
    for ipynb_file in ipynb_files:
        h5_file = ipynb_file.replace('.ipynb', '.h5')
        h5_path = os.path.join(current_dir, h5_file)
        ipynb_path = os.path.join(current_dir, ipynb_file)
        
        if os.path.exists(h5_path):
            print(f"\n{'='*60}")
            print(f"Checking: {ipynb_file} ↔ {h5_file}")
            print(f"{'='*60}")
            
            # Check H5 structure
            check_h5_structure(h5_path)
            
            # Detailed comparison
            is_correct = detailed_verification(ipynb_path, h5_path)
            
            if is_correct:
                print(f"\n✅ {h5_file} conversion is CORRECT")
            else:
                print(f"\n❌ {h5_file} conversion has ISSUES")
        else:
            print(f"\n⚠️  No H5 file found for {ipynb_file}")

if __name__ == "__main__":
    main()
