import os

IGNORE_FOLDERS = {"venv", ".git", ".idea"}  # add more if needed

for root, dirs, files in os.walk(".", topdown=True):
    # Remove ignored directories from traversal
    dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

    level = root.replace(os.getcwd(), '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
