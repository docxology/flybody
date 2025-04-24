#!/usr/bin/env python3
"""
Parses Python files in a specified directory (and subdirectories)
to extract information about functions defined within them.

Uses Python's Abstract Syntax Trees (AST) module.
Outputs the extracted information in JSON format to standard output.
"""

import os
import ast
import json
import sys
from typing import List, Dict, Any, Optional

# The target directory containing the Python source code.
# Corresponds to the @flybody alias used in the request.
SOURCE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../flybody')
)


def unparse_annotation(node: Optional[ast.expr]) -> Optional[str]:
    """Safely attempts to unparse an AST node representing a type annotation."""
    if node is None:
        return None
    try:
        # Use ast.unparse() available in Python 3.9+
        # For older versions, a more complex recursive approach or
        # a third-party library like 'astor' would be needed.
        if hasattr(ast, 'unparse'):
            return ast.unparse(node)
        else:
            # Basic fallback for common types, not comprehensive
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                # Simple attribute access like x.y
                return f"{unparse_annotation(node.value)}.{node.attr}"
            if isinstance(node, ast.Subscript):
                 # Basic subscript like List[int]
                 value = unparse_annotation(node.value)
                 slice_val = unparse_annotation(node.slice)
                 return f"{value}[{slice_val}]"
            return "<complex_annotation>" # Placeholder for unhandled types
    except Exception:
        return "<unparse_error>"

def extract_function_info(filepath: str) -> List[Dict[str, Any]]:
    """Extracts function information from a single Python file."""
    functions_info = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
            tree = ast.parse(source_code, filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info: Dict[str, Any] = {
                        'name': node.name,
                        'args': [],
                        'return_annotation': unparse_annotation(node.returns),
                        'docstring': ast.get_docstring(node),
                    }
                    # Extract arguments and their annotations
                    for arg in node.args.args:
                        arg_info = {
                            'name': arg.arg,
                            'annotation': unparse_annotation(arg.annotation)
                        }
                        func_info['args'].append(arg_info)

                    # Handle *args
                    if node.args.vararg:
                         func_info['vararg'] = {
                            'name': node.args.vararg.arg,
                            'annotation': unparse_annotation(node.args.vararg.annotation)
                         }

                    # Handle **kwargs
                    if node.args.kwarg:
                         func_info['kwarg'] = {
                            'name': node.args.kwarg.arg,
                            'annotation': unparse_annotation(node.args.kwarg.annotation)
                         }

                    # Handle keyword-only args
                    func_info['kwonlyargs'] = []
                    for arg in node.args.kwonlyargs:
                        arg_info = {
                            'name': arg.arg,
                            'annotation': unparse_annotation(arg.annotation)
                        }
                        func_info['kwonlyargs'].append(arg_info)

                    functions_info.append(func_info)

    except FileNotFoundError:
        print(f"Error: File not found {filepath}", file=sys.stderr)
    except SyntaxError as e:
        print(f"Error: Could not parse {filepath} - {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: An unexpected error occurred processing {filepath}: {e}", file=sys.stderr)

    return functions_info

def scan_directory(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Scans the directory for .py files and extracts function info from each."""
    all_files_info = {}
    print(f"Scanning for Python files in: {directory}\n")

    for root, _, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue

        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, directory)
                print(f"Processing: {relative_path}")
                functions = extract_function_info(filepath)
                if functions:
                    all_files_info[relative_path] = functions

    return all_files_info

if __name__ == "__main__":
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    extracted_data = scan_directory(SOURCE_DIR)

    if not extracted_data:
        print("\nNo function information extracted.")
        sys.exit(0)

    print("\nExtraction complete. Outputting JSON:")
    try:
        json_output = json.dumps(extracted_data, indent=2)
        print(json_output)
        # Optionally write to a file:
        # with open('function_info.json', 'w') as f_out:
        #     f_out.write(json_output)
        # print("\nJSON output also saved to function_info.json")
        sys.exit(0)
    except TypeError as e:
        print(f"\nError: Could not serialize extracted data to JSON: {e}", file=sys.stderr)
        sys.exit(1) 