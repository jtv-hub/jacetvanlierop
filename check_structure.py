"""
Utility to print the directory structure of the 'src' folder for debugging.
"""
import os

def print_directory_structure(startpath, max_depth=3):
    """
    Recursively prints the directory structure starting from the given path
    up to the specified depth.

    Args:
        startpath (str): The root directory to start printing from.
        max_depth (int): The maximum depth to traverse in the directory structure.
    """
    for root, _, files in os.walk(startpath):
        # Limit depth
        depth = root.replace(startpath, '').count(os.sep)
        if depth >= max_depth:
            continue

        indent = '│   ' * depth + '├── '
        print(f"{indent}{os.path.basename(root)}/")

        for f in files:
            print(f"{indent}    {f}")

if __name__ == "__main__":
    print("📂 Directory Structure (3 levels deep):\n")
    print_directory_structure("src")
