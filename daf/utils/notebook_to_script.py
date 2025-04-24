#!/usr/bin/env python3
"""
Converts all Jupyter Notebook (.ipynb) files found in the specified directory
(and its subdirectories) into Python (.py) scripts in place.

This script relies on the 'jupyter nbconvert' command-line tool.
Ensure you have Jupyter installed (`pip install jupyter`).
"""

import os
import subprocess
import sys

# The target directory containing the notebooks.
# Corresponds to the @docs alias used in the request.
NOTEBOOK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../docs')
)

def convert_notebooks_to_scripts(directory: str):
    """
    Walks through the directory and converts .ipynb files to .py using nbconvert.
    """
    converted_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Searching for notebooks in: {directory}\n")

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".ipynb"):
                notebook_path = os.path.join(root, filename)
                script_path = os.path.splitext(notebook_path)[0] + ".py"
                print(f"Processing: {notebook_path}")

                # Check if the script already exists - skip if it does
                # if os.path.exists(script_path):
                #     print(f"  Skipping: Corresponding script '{script_path}' already exists.")
                #     skipped_count += 1
                #     continue

                try:
                    # Use --inplace to modify the file type directly if supported
                    # Using --to script writes a separate .py file
                    # Using --stdout redirects, which isn't what we want here.
                    # We will write the .py file alongside the .ipynb
                    command = [
                        sys.executable,  # Use the current python interpreter to find jupyter
                        "-m",
                        "jupyter",
                        "nbconvert",
                        "--to",
                        "script",
                        notebook_path
                    ]
                    print(f"  Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True, check=False)

                    if result.returncode == 0:
                        print(f"  Successfully converted to: {script_path}")
                        converted_count += 1
                    else:
                        print(f"  Error converting {notebook_path}:", file=sys.stderr)
                        print(result.stderr, file=sys.stderr)
                        error_count += 1

                except FileNotFoundError:
                    print("Error: 'jupyter' command not found.", file=sys.stderr)
                    print("Please ensure Jupyter is installed and in your PATH ('pip install jupyter').", file=sys.stderr)
                    return False
                except Exception as e:
                    print(f"An unexpected error occurred for {notebook_path}: {e}", file=sys.stderr)
                    error_count += 1

    print("\nConversion Summary:")
    print(f"  Converted: {converted_count}")
    # print(f"  Skipped (already exist): {skipped_count}") # Removed skip logic
    print(f"  Errors:    {error_count}")
    return error_count == 0

if __name__ == "__main__":
    if not os.path.isdir(NOTEBOOK_DIR):
        print(f"Error: Target directory '{NOTEBOOK_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    if not convert_notebooks_to_scripts(NOTEBOOK_DIR):
        print("\nConversion process completed with errors.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nConversion process completed successfully.")
        sys.exit(0) 