import os
import sys

# Directories to ignore (case-insensitive)
IGNORE_DIRS = {
    '.venv', 'venv', '.vscode', 'node_modules', '__pycache__', '.git',
    '.idea', '.mypy_cache', '.pytest_cache', 'build', 'dist', '.tox'
}

def should_ignore(name: str) -> bool:
    return name.lower() in IGNORE_DIRS or name.startswith('.')

def print_tree(root_dir: str = ".", prefix: str = "") -> None:
    """Recursively print the directory tree."""
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    entries = sorted(os.listdir(root_dir))
    # Filter out ignored directories and hidden files/folders (unless explicitly needed)
    filtered_entries = []
    for entry in entries:
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            if should_ignore(entry):
                continue
        filtered_entries.append(entry)

    total = len(filtered_entries)
    for i, entry in enumerate(filtered_entries):
        path = os.path.join(root_dir, entry)
        is_last = (i == total - 1)
        connector = "└─ " if is_last else "├─ "

        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "   " if is_last else "│  "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    # Use the first command-line argument as root, or default to current dir
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    project_name = os.path.basename(os.path.abspath(root))
    print(f"{project_name}/")
    print_tree(root)