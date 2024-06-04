import nbformat
import argparse
from pathlib import Path

def export_code_cells(nb_path, output_path):
    """
    Exports code cells that start with '#export' from a Jupyter notebook to a Python file.
    The '#export' line is removed from the exported code.

    Args:
    nb_path (str): Path to the Jupyter notebook file.
    output_path (str): Path where the output Python file should be saved.
    """
    # Load the notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Collect code cells that start with '#export'
    code_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.source.startswith('#export'):
            # Remove the '#export' line
            modified_source = '\n'.join(line for line in cell.source.split('\n') if not line.strip().startswith('#export'))
            code_cells.append(modified_source)

    # Write the selected code cells to the output Python file
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for cell in code_cells:
            f.write(cell + '\n\n')

def main():
    parser = argparse.ArgumentParser(description="Export code cells from a Jupyter notebook to a Python file, excluding '#export' tags.")
    parser.add_argument("notebook_path", type=str, help="Path to the Jupyter notebook.")
    parser.add_argument("output_path", type=str, help="Path to save the output Python file.")
    
    args = parser.parse_args()

    export_code_cells(args.notebook_path, args.output_path)

if __name__ == "__main__":
    main()
