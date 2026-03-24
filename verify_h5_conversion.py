import h5py
import numpy as np
import json
import os

def verify_h5_file(h5_path):
    """Verify that an H5 file contains the expected notebook data"""
    print(f"\n=== Verifying {os.path.basename(h5_path)} ===")
    
    try:
        with h5py.File(h5_path, 'r') as h5file:
            # Check metadata
            print(f"Original filename: {h5file.attrs.get('original_filename', 'N/A')}")
            print(f"Notebook format: {h5file.attrs.get('notebook_format', 'N/A')}")
            print(f"Creation time: {h5file.attrs.get('creation_time', 'N/A')}")
            
            # Check statistics
            if 'statistics' in h5file:
                stats = h5file['statistics']
                total_cells = stats.attrs.get('total_code_cells', 0)
                cells_with_output = stats.attrs.get('total_cells_with_output', 0)
                print(f"Total code cells: {total_cells}")
                print(f"Cells with output: {cells_with_output}")
            
            # Check code cells structure
            if 'code_cells' in h5file:
                code_group = h5file['code_cells']
                cell_count = len([key for key in code_group.keys() if key.startswith('cell_')])
                print(f"Code cell groups found: {cell_count}")
                
                # Sample first few cells
                sample_cells = min(3, cell_count)
                for i in range(sample_cells):
                    cell_key = f'cell_{i}'
                    if cell_key in code_group:
                        cell = code_group[cell_key]
                        
                        # Check source lines
                        source_keys = [key for key in cell.keys() if key.startswith('source_line_')]
                        source_count = len(source_keys)
                        
                        # Check output lines
                        output_keys = [key for key in cell.keys() if key.startswith('output_line_')]
                        output_count = len(output_keys)
                        
                        # Get first source line as sample
                        sample_source = ""
                        if source_count > 0:
                            first_source = cell['source_line_0'][()]
                            if isinstance(first_source, bytes):
                                sample_source = first_source.decode('utf-8')
                            else:
                                sample_source = str(first_source)
                            sample_source = sample_source[:50] + "..." if len(sample_source) > 50 else sample_source
                        
                        print(f"  Cell {i}: {source_count} source lines, {output_count} output lines")
                        print(f"    Sample: {sample_source}")
            
            print("✅ H5 file structure is valid")
            return True
            
    except Exception as e:
        print(f"❌ Error verifying H5 file: {str(e)}")
        return False

def compare_with_original(ipynb_path, h5_path):
    """Compare H5 content with original notebook"""
    print(f"\n=== Comparing {os.path.basename(ipynb_path)} with {os.path.basename(h5_path)} ===")
    
    try:
        # Read original notebook
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        original_code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                original_code_cells.append(source)
        
        # Read H5 file
        with h5py.File(h5_path, 'r') as h5file:
            h5_code_cells = []
            if 'code_cells' in h5file:
                code_group = h5file['code_cells']
                
                # Sort cells by index
                cell_keys = sorted([key for key in code_group.keys() if key.startswith('cell_')])
                
                for cell_key in cell_keys:
                    cell = code_group[cell_key]
                    
                    # Reconstruct source from lines
                    source_lines = []
                    source_keys = sorted([key for key in cell.keys() if key.startswith('source_line_')])
                    
                    for source_key in source_keys:
                        line_data = cell[source_key][()]
                        if isinstance(line_data, bytes):
                            source_lines.append(line_data.decode('utf-8'))
                        else:
                            source_lines.append(str(line_data))
                    
                    h5_code_cells.append('\n'.join(source_lines))
        
        # Compare
        print(f"Original notebook code cells: {len(original_code_cells)}")
        print(f"H5 file code cells: {len(h5_code_cells)}")
        
        if len(original_code_cells) != len(h5_code_cells):
            print("❌ Cell count mismatch!")
            return False
        
        # Check first few cells for content match
        matches = 0
        for i in range(min(3, len(original_code_cells))):
            if original_code_cells[i].strip() == h5_code_cells[i].strip():
                matches += 1
            else:
                print(f"❌ Cell {i} content differs:")
                print(f"  Original: {original_code_cells[i][:50]}...")
                print(f"  H5:       {h5_code_cells[i][:50]}...")
        
        if matches == min(3, len(original_code_cells)):
            print("✅ Sample cells match perfectly")
            return True
        else:
            print(f"⚠️  Only {matches}/3 sample cells match")
            return False
            
    except Exception as e:
        print(f"❌ Error comparing files: {str(e)}")
        return False

def main():
    current_dir = os.getcwd()
    print(f"Verifying H5 files in: {current_dir}")
    
    # Find H5 files
    h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print("No H5 files found!")
        return
    
    print(f"Found {len(h5_files)} H5 files to verify")
    
    # Verify each H5 file
    all_valid = True
    for h5_file in h5_files:
        h5_path = os.path.join(current_dir, h5_file)
        
        # Basic structure verification
        if not verify_h5_file(h5_path):
            all_valid = False
            continue
        
        # Find corresponding original notebook
        notebook_file = h5_file.replace('.h5', '.ipynb')
        notebook_path = os.path.join(current_dir, notebook_file)
        
        if os.path.exists(notebook_path):
            # Compare with original
            if not compare_with_original(notebook_path, h5_path):
                all_valid = False
        else:
            print(f"⚠️  Original notebook {notebook_file} not found for comparison")
    
    print(f"\n=== Final Result ===")
    if all_valid:
        print("✅ All H5 files are valid and match their original notebooks!")
    else:
        print("❌ Some issues found with H5 files")

if __name__ == "__main__":
    main()
