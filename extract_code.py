import os
import json

def extract_code_from_folder(root_path, output_file='extracted_code.txt'):
    """
    Extract code from all files in a folder structure and save to a text file.
    
    Args:
        root_path: Root directory path to scan
        output_file: Output file name
    """
    
    # File extensions to include
    code_extensions = {
        '.py','.env', '.yml', '.yaml'
    }
    
    # Files/folders to exclude
    exclude_dirs = {'node_modules', '.git', 'dist', 'build', '__pycache__', 'rag_venv', '.venv'}
    exclude_files = {
        'package-lock.json', '.DS_Store',
        'extract_code.py', 'project_structure.txt',
        'rag_requirements.txt', 'requirements.txt', 'structure.py', 
        '.gitignore', 'GatedFusion_test_predictions.json', 
        'training_histories.json', 'training_histories_kaggle.json', 
        'lstm_training_histories.json', 'kaggle_session_tracking.json', 
        'fraud_cases_database.json', 'elliptic_id2idx.json', 
        'UltraDeepSAGE_L8_H352_history.json', 
        'HybridGNN_L4_H320_history.json', 
        'DeepSAGE_L7_H400_history.json', 'DeepSAGE_L6_H384_history.json'
    }
    
    # Files that should not be displayed (sensitive)
    sensitive_files = {'.env', 'firebase-adminsdk'}
    
    # Files with no extensions but should be included
    special_files = {'Dockerfile', 'docker-compose.yml'}
    
    extracted_files = []
    
    print(f"Scanning directory: {root_path}")
    print("=" * 80)
    
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove excluded directories from the walk
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        # Get relative path
        rel_dir = os.path.relpath(dirpath, root_path)
        
        for filename in filenames:
            # Skip excluded files
            if filename in exclude_files:
                continue
            
            # Get file extension
            _, ext = os.path.splitext(filename)
            
            # Check if file should be processed
            if ext in code_extensions or filename in special_files or filename in ['.gitignore', '.env.example']:
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.join(rel_dir, filename).replace('\\', '/')
                
                # Check if sensitive file
                is_sensitive = any(sens in filename for sens in sensitive_files)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Store file info
                    extracted_files.append({
                        'path': rel_path,
                        'content': content if not is_sensitive else '[SENSITIVE - CONTENT HIDDEN]',
                        'is_sensitive': is_sensitive,
                        'size': len(content)
                    })
                    
                    print(f"✓ Extracted: {rel_path}")
                    
                except Exception as e:
                    print(f"✗ Error reading {rel_path}: {str(e)}")
    
    # Write to output file
    print("\n" + "=" * 80)
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CODE EXTRACTION FROM: {root_path}\n")
        f.write(f"Total files extracted: {len(extracted_files)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort files by path
        extracted_files.sort(key=lambda x: x['path'])
        
        for file_info in extracted_files:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"FILE: {file_info['path']}\n")
            f.write("=" * 80 + "\n")
            
            if file_info['is_sensitive']:
                f.write("[SENSITIVE FILE - CONTENT NOT DISPLAYED FOR SECURITY]\n")
                f.write("This file may contain passwords, API keys, or credentials.\n")
            else:
                f.write(file_info['content'])
                if not file_info['content'].endswith('\n'):
                    f.write('\n')
            
            f.write("\n")
    
    print(f"✓ Extraction complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Total files: {len(extracted_files)}")
    print(f"✓ Total size: {sum(f['size'] for f in extracted_files):,} characters")


def create_project_structure(root_path, output_file='project_structure.txt'):
    """
    Create a visual tree structure of the project.
    
    Args:
        root_path: Root directory path to scan
        output_file: Output file name
    """
    exclude_dirs = {'node_modules', '.git', 'dist', 'build', '__pycache__', 'rag_venv', '.venv'}
    
    tree_lines = []
    
    def add_tree_line(path, prefix="", is_last=True):
        name = os.path.basename(path)
        connector = "└── " if is_last else "├── "
        tree_lines.append(prefix + connector + name)
        
        if os.path.isdir(path):
            items = sorted(os.listdir(path))
            items = [item for item in items if item not in exclude_dirs]
            
            for i, item in enumerate(items):
                item_path = os.path.join(path, item)
                is_last_item = (i == len(items) - 1)
                extension = "    " if is_last else "│   "
                add_tree_line(item_path, prefix + extension, is_last_item)
    
    tree_lines.append(os.path.basename(root_path) + "/")
    
    items = sorted(os.listdir(root_path))
    items = [item for item in items if item not in exclude_dirs]
    
    for i, item in enumerate(items):
        item_path = os.path.join(root_path, item)
        is_last = (i == len(items) - 1)
        add_tree_line(item_path, "", is_last)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PROJECT STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        f.write('\n'.join(tree_lines))
    
    print(f"✓ Project structure saved to: {output_file}")


if __name__ == "__main__":
    # Set your project path here
    project_path = r"C:\Users\youss\Downloads\Flag_finance"
    
    print("CODE EXTRACTOR")
    print("=" * 80)
    print()
    
    # Check if path exists
    if not os.path.exists(project_path):
        print(f"Error: Path does not exist: {project_path}")
        print("Please update the 'project_path' variable in the script.")
    else:
        # Create project structure
        create_project_structure(project_path, 'project_structure.txt')
        print()
        
        # Extract all code
        extract_code_from_folder(project_path, 'extracted_code.txt')
        print()
        print("=" * 80)
        print("DONE! Check these files:")
        print("  - extracted_code.txt (all code content)")
        print("  - project_structure.txt (directory tree)")
        print("=" * 80)
