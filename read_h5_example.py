import h5py
import os

def read_h5_notebook_example(h5_path):
    """Example of how to read the notebook data back from H5 file"""
    print(f"\n=== Reading {os.path.basename(h5_path)} ===")
    
    with h5py.File(h5_path, 'r') as h5file:
        print(f"Original filename: {h5file.attrs.get('original_filename')}")
        print(f"Created: {h5file.attrs.get('creation_time')}")
        
        # Get statistics
        if 'statistics' in h5file:
            stats = h5file['statistics']
            total_cells = stats.attrs.get('total_code_cells')
            cells_with_output = stats.attrs.get('total_cells_with_output')
            print(f"Total cells: {total_cells}, with output: {cells_with_output}")
        
        # Read first few cells as example
        if 'code_cells' in h5file:
            code_group = h5file['code_cells']
            cell_keys = sorted([key for key in code_group.keys() if key.startswith('cell_')])
            
            print(f"\n--- First 3 Code Cells ---")
            for i, cell_key in enumerate(cell_keys[:3]):
                cell = code_group[cell_key]
                
                # Reconstruct source code
                source_lines = []
                source_keys = sorted([key for key in cell.keys() if key.startswith('source_line_')])
                
                for source_key in source_keys:
                    line_data = cell[source_key][()]
                    if isinstance(line_data, bytes):
                        source_lines.append(line_data.decode('utf-8'))
                    else:
                        source_lines.append(str(line_data))
                
                source_code = '\n'.join(source_lines)
                
                # Get execution info
                exec_count = cell.attrs.get('execution_count', 'N/A')
                has_output = cell.attrs.get('has_output', False)
                output_count = cell.attrs.get('outputs_count', 0)
                
                print(f"\nCell {i} (exec_count: {exec_count}, outputs: {output_count}):")
                print("-" * 40)
                # Show first few lines
                lines = source_code.split('\n')
                for line in lines[:5]:
                    print(line)
                if len(lines) > 5:
                    print(f"... ({len(lines)-5} more lines)")

def main():
    current_dir = os.getcwd()
    h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]
    
    print("H5 File Reading Demonstration")
    print("=" * 50)
    
    for h5_file in h5_files:
        h5_path = os.path.join(current_dir, h5_file)
        read_h5_notebook_example(h5_path)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
