import json
import h5py
import numpy as np
from datetime import datetime
import os
import re

def extract_code_from_notebook(notebook_path):
    """Extract code cells from a Jupyter notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    metadata = {
        'notebook_format': notebook.get('nbformat', 0),
        'notebook_format_minor': notebook.get('nbformat_minor', 0),
        'metadata': notebook.get('metadata', {}),
        'creation_time': datetime.now().isoformat()
    }
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            
            # Extract code and output information
            cell_data = {
                'source': source,
                'execution_count': cell.get('execution_count'),
                'outputs_count': len(outputs),
                'has_output': len(outputs) > 0
            }
            
            # Store output text if available
            output_text = []
            for output in outputs:
                if 'text' in output:
                    if isinstance(output['text'], list):
                        output_text.extend(output['text'])
                    else:
                        output_text.append(output['text'])
                elif 'data' in output:
                    for data_type, data_content in output['data'].items():
                        if data_type == 'text/plain':
                            if isinstance(data_content, list):
                                output_text.extend(data_content)
                            else:
                                output_text.append(data_content)
            
            cell_data['output_text'] = '\n'.join(output_text)
            code_cells.append(cell_data)
    
    return metadata, code_cells

def notebook_to_h5(notebook_path, output_path):
    """Convert a Jupyter notebook to HDF5 format"""
    print(f"Converting {notebook_path} to {output_path}")
    
    # Extract data from notebook
    metadata, code_cells = extract_code_from_notebook(notebook_path)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5file:
        # Store metadata as attributes
        for key, value in metadata.items():
            if key == 'metadata':
                # Store nested metadata as JSON string
                h5file.attrs[key] = json.dumps(value)
            else:
                h5file.attrs[key] = str(value)
        
        # Create groups for organization
        code_group = h5file.create_group('code_cells')
        
        # Store each code cell
        for i, cell in enumerate(code_cells):
            cell_group = code_group.create_group(f'cell_{i}')
            
            # Store source code
            if cell['source']:
                # Split source into lines for better storage
                source_lines = cell['source'].split('\n')
                for j, line in enumerate(source_lines):
                    cell_group.create_dataset(f'source_line_{j}', data=line.encode('utf-8'))
            
            # Store execution count if available
            if cell['execution_count'] is not None:
                cell_group.attrs['execution_count'] = cell['execution_count']
            
            # Store output information
            cell_group.attrs['outputs_count'] = cell['outputs_count']
            cell_group.attrs['has_output'] = cell['has_output']
            
            # Store output text if available
            if cell['output_text']:
                output_lines = cell['output_text'].split('\n')
                for j, line in enumerate(output_lines):
                    cell_group.create_dataset(f'output_line_{j}', data=line.encode('utf-8'))
        
        # Store summary statistics
        stats_group = h5file.create_group('statistics')
        stats_group.attrs['total_code_cells'] = len(code_cells)
        stats_group.attrs['total_cells_with_output'] = sum(1 for cell in code_cells if cell['has_output'])
        
        # Store notebook name
        notebook_name = os.path.basename(notebook_path)
        h5file.attrs['original_filename'] = notebook_name
        
        print(f"  - Stored {len(code_cells)} code cells")
        print(f"  - {sum(1 for cell in code_cells if cell['has_output'])} cells have outputs")

def convert_all_notebooks(directory):
    """Convert all .ipynb files in a directory to .h5 format"""
    notebook_files = [f for f in os.listdir(directory) if f.endswith('.ipynb')]
    
    if not notebook_files:
        print("No .ipynb files found in the directory.")
        return
    
    print(f"Found {len(notebook_files)} notebook files:")
    for notebook in notebook_files:
        print(f"  - {notebook}")
    
    print("\nStarting conversion...")
    
    for notebook_file in notebook_files:
        notebook_path = os.path.join(directory, notebook_file)
        h5_filename = notebook_file.replace('.ipynb', '.h5')
        h5_path = os.path.join(directory, h5_filename)
        
        try:
            notebook_to_h5(notebook_path, h5_path)
            print(f"✓ Successfully converted {notebook_file} to {h5_filename}")
        except Exception as e:
            print(f"✗ Error converting {notebook_file}: {str(e)}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Converting notebooks in: {current_dir}")
    
    convert_all_notebooks(current_dir)
