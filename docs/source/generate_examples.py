import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.abspath("../.."))

# ==== CONFIGURATION ====
examples_root = "../../examples"
output_root = "./examples"
root_index_file = "./examples_index.md"
main_index_file = "./index.md"
os.makedirs(output_root, exist_ok=True)


# ==== GET GITHUB URL ====
def get_github_base_url():
    try:
        url = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        url = url.replace("git@github.com:", "https://github.com/")
        url = url.replace(".git", "")
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return f"{url}/blob/{branch}/examples"
    except Exception:
        return None


# ==== FORMAT TITLE ====
def format_title(path: str):
    return " / ".join(
        part.replace("_", " ").title() for part in path.split(os.sep) if part
    )


GITHUB_BASE_URL = get_github_base_url()

# ==== GET FILES ====
folder_to_files = defaultdict(list)

for root, dirs, files in os.walk(examples_root):
    for filename in sorted(files):
        if filename.endswith(".py"):
            relative_dir = os.path.relpath(root, examples_root)
            output_dir = os.path.join(output_root, relative_dir)
            os.makedirs(output_dir, exist_ok=True)

            example_name = filename[:-3]
            md_filename = f"{example_name}.md"
            md_filepath = os.path.join(output_dir, md_filename)

            py_relative_path = os.path.join("examples", relative_dir, filename).replace(
                "\\", "/"
            )
            github_url = (
                f"{GITHUB_BASE_URL}/{relative_dir}/{filename}".replace("\\", "/")
                if GITHUB_BASE_URL
                else None
            )

            folder_to_files[relative_dir].append(md_filename)

            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(f"# `{example_name}.py`\n\n")
                f.write(f"**Source**: [`{py_relative_path}`]({github_url})  \n")

                f.write("```python\n")
                with open(
                    os.path.join(root, filename), "r", encoding="utf-8"
                ) as py_file:
                    f.write(py_file.read())
                f.write("\n```\n")

# ==== WRITE INDEX.MD FOR FOLDERS ====
for relative_dir, files in folder_to_files.items():
    output_dir = os.path.join(output_root, relative_dir)
    index_md_path = os.path.join(output_dir, "index.md")
    title = format_title(relative_dir) if relative_dir != "." else "Examples Index"

    with open(index_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("```{toctree}\n")
        f.write(":maxdepth: 1\n\n")
        for file in sorted(files):
            f.write(f"{file}\n")
        f.write("```\n")

# ==== WRITE INDEX.MD ====
top_level_folders = set(
    [path.split(os.sep)[0] for path in folder_to_files if path != "."]
)

with open(root_index_file, "w", encoding="utf-8") as f:
    f.write("# Examples Usage\n\n")
    f.write("```{toctree}\n")
    f.write(":maxdepth: 2\n\n")
    for file in folder_to_files.get(".", []):
        f.write(f"examples/{file}\n")
    for folder in sorted(top_level_folders):
        f.write(f"examples/{folder}/index.md\n")
    f.write("```\n")

# ==== GENERATE INDEX.MD FOR SUBFOLDERS ====
all_index_paths = []
for root, dirs, files in os.walk(output_root):
    for file in files:
        if file == "index.md":
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, output_root)
            all_index_paths.append(Path(relative_path))

parent_to_children = defaultdict(list)
for path in all_index_paths:
    if path.parent != Path("."):
        parent_to_children[path.parent.parent].append(path.parent)

for parent, children in parent_to_children.items():
    index_path = Path(output_root) / parent / "index.md"
    if not index_path.exists():
        os.makedirs(index_path.parent, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# {format_title(str(parent))}\n\n")
            f.write("```{toctree}\n")
            f.write(":maxdepth: 1\n\n")
            for child in sorted(children):
                f.write(f"{child.name}/index.md\n")
            f.write("```\n")

if "." not in folder_to_files:
    try:
        os.remove(os.path.join(output_root, "index.md"))
    except FileNotFoundError:
        pass


# ==== ADD EXAMPLES INDEX TO MAIN INDEX ====
def ensure_examples_in_main_index():
    if not os.path.exists(main_index_file):
        return
    with open(main_index_file, "r", encoding="utf-8") as f:
        content = f.read()

    if "examples_index.md" in content:
        return

    new_content = (
        content
        + "\n\n```{toctree}\n:maxdepth: 2\n:caption: Examples\n\nexamples_index.md\n```\n"
    )
    with open(main_index_file, "w", encoding="utf-8") as f:
        f.write(new_content)


ensure_examples_in_main_index()
