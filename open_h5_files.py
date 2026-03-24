import h5py
import os

def open_and_display_h5_file(h5_path):
    """Open and display complete contents of an H5 file"""
    print(f"\n{'='*80}")
    print(f"📁 OPENING: {os.path.basename(h5_path)}")
    print(f"{'='*80}")
    
    with h5py.File(h5_path, 'r') as h5file:
        # Display file attributes
        print(f"\n📋 FILE ATTRIBUTES:")
        print("-" * 40)
        for attr_name, attr_value in h5file.attrs.items():
            print(f"{attr_name}: {attr_value}")
        
        # Display statistics
        if 'statistics' in h5file:
            print(f"\n📊 STATISTICS:")
            print("-" * 40)
            stats = h5file['statistics']
            for attr_name, attr_value in stats.attrs.items():
                print(f"{attr_name}: {attr_value}")
        
        # Display all code cells
        if 'code_cells' in h5file:
            print(f"\n💻 CODE CELLS:")
            print("-" * 40)
            code_group = h5file['code_cells']
            
            # Get cells in order
            cell_keys = sorted([key for key in code_group.keys() if key.startswith('cell_')])
            
            for i, cell_key in enumerate(cell_keys):
                cell = code_group[cell_key]
                print(f"\n🔹 CELL {i}: {cell_key}")
                print("   " + "-" * 50)
                
                # Display cell attributes
                exec_count = cell.attrs.get('execution_count', 'N/A')
                outputs_count = cell.attrs.get('outputs_count', 0)
                has_output = cell.attrs.get('has_output', False)
                
                print(f"   Execution Count: {exec_count}")
                print(f"   Outputs Count: {outputs_count}")
                print(f"   Has Output: {has_output}")
                
                # Reconstruct and display source code
                source_keys = sorted([key for key in cell.keys() if key.startswith('source_line_')])
                if source_keys:
                    print(f"   📝 SOURCE CODE ({len(source_keys)} lines):")
                    source_lines = []
                    for source_key in source_keys:
                        line_data = cell[source_key][()]
                        if isinstance(line_data, bytes):
                            source_lines.append(line_data.decode('utf-8'))
                        else:
                            source_lines.append(str(line_data))
                    
                    source_code = '\n'.join(source_lines)
                    
                    # Display first 10 lines, then summary
                    lines = source_code.split('\n')
                    for j, line in enumerate(lines[:10]):
                        print(f"   {j+1:2d}: {line}")
                    
                    if len(lines) > 10:
                        print(f"   ... ({len(lines)-10} more lines)")
                        print(f"   Total characters: {len(source_code)}")
                
                # Display output summary
                output_keys = sorted([key for key in cell.keys() if key.startswith('output_line_')])
                if output_keys:
                    print(f"   📤 OUTPUT ({len(output_keys)} lines):")
                    # Show first 3 lines of output
                    for j in range(min(3, len(output_keys))):
                        output_key = f'output_line_{j}'
                        if output_key in cell:
                            output_data = cell[output_key][()]
                            if isinstance(output_data, bytes):
                                output_line = output_data.decode('utf-8')
                            else:
                                output_line = str(output_data)
                            print(f"   {j+1}: {output_line[:80]}{'...' if len(output_line) > 80 else ''}")
                    
                    if len(output_keys) > 3:
                        print(f"   ... ({len(output_keys)-3} more output lines)")

def main():
    current_dir = os.getcwd()
    h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]
    
    print("🔍 OPENING ALL H5 FILES...")
    print(f"Found {len(h5_files)} H5 files to open")
    
    for h5_file in sorted(h5_files):
        h5_path = os.path.join(current_dir, h5_file)
        open_and_display_h5_file(h5_path)
        
        # Ask if user wants to continue to next file
        if h5_file != sorted(h5_files)[-1]:  # Not the last file
            print(f"\n{'='*80}")
            input("Press Enter to continue to the next file...")
    
    print(f"\n{'='*80}")
    print("✅ ALL H5 FILES DISPLAYED COMPLETELY!")

if __name__ == "__main__":
    main()
