import os

def display_directory_structure(root_dir, indent=""):
    """
    Recursively print the directory structure starting from `root_dir`.
    """
    if not os.path.exists(root_dir):
        print(f"The directory '{root_dir}' does not exist.")
        return
    
    items = sorted(os.listdir(root_dir))  # Sort items for consistent order
    for item in items:
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            display_directory_structure(item_path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

# Set the root directory to inspect (change "." to your project's path if needed)
root_directory = "."
display_directory_structure(root_directory)
