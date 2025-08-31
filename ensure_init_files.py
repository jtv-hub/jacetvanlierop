"""
Utility script to ensure every folder in the project has an __init__.py file.
This makes them valid Python packages for absolute imports.
"""

import os

def ensure_init_files(root_folder: str):
    """Walk through the project and ensure every folder has an __init__.py"""
    for dirpath, _, _ in os.walk(root_folder):
        # Skip virtualenv and hidden folders
        if "venv" in dirpath or "/." in dirpath:
            continue

        init_file = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# Auto-created to make this a Python package\n")
            print(f"[added] {init_file}")
        else:
            print(f"[exists] {init_file}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    ensure_init_files(project_root)
