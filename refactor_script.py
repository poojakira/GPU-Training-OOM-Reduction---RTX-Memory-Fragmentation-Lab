import os
import fnmatch

def replace_in_file(file_path, old_str, new_str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_str in content:
            new_content = content.replace(old_str, new_str)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False

def main():
    root_dir = os.getcwd()
    patterns = ['*.py', '*.md', '*.yaml', '*.toml', '*.js', '*.jsx', '*.html', '*.json', '*.txt']
    exclude_dirs = {'.git', '.pytest_cache', '__pycache__', 'node_modules', '.ruff_cache', 'dist'}

    replacements = [
        ('apex_aegis', 'apex_aegis'),
        ('apex-aegis', 'apex-aegis'),
    ]

    count = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in files:
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    file_path = os.path.join(root, filename)
                    changed = False
                    for old_s, new_s in replacements:
                        if replace_in_file(file_path, old_s, new_s):
                            changed = True
                    if changed:
                        count += 1
                        print(f"Updated: {file_path}")
    
    print(f"Total files updated: {count}")

if __name__ == "__main__":
    main()
