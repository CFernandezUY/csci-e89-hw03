import ast, glob, json, os, re, uuid

LAST, FIRST, PROB = "Fernandez", "Carlos", 4
TOOL = "OpenAI Codex"

def md(t):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": t.splitlines(True)}

def code(t):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": t.splitlines(True)}

def add_file_var(src, path):
    """Notebooks have no __file__; define it for scripts that use it."""
    if "__file__" not in src:
        return src
    line = f'__file__ = r"{path}"  # notebooks have no __file__, so define it here\n'
    future = list(re.finditer(r"^from __future__ import .*\n", src, re.M))
    if future:
        i = future[-1].end()
        return src[:i] + line + src[i:]
    return line + src

cells = [
    md(f"# CSCI E-89 Assignment 03 - Problem {PROB}\n### Author: {FIRST} {LAST}\n\n"
       f"Code generated with {TOOL}. Each code cell is labelled with the script it comes from."),
    code("import sys\nsys.path.insert(0, 'scripts')"),
]
for path in sorted(glob.glob("scripts/*.py")):
    src = open(path, encoding="utf-8").read()
    doc = ast.get_docstring(ast.parse(src)) or "(add: what this script does and why)"
    cells.append(md(f"## {os.path.basename(path)}\n\n**What was done and why:** {doc}"))
    cells.append(code(f"# ===== Notebook cell for generated script: {path} =====\n"
                      + add_file_var(src, path.replace(os.sep, "/"))))
    if "savefig" in src:
        cells.append(md(f"Plot saved by `{os.path.basename(path)}`, displayed here so it appears in the notebook and the HTML export:"))
        cells.append(code("from IPython.display import Image, display\n"
                          "import glob\n"
                          "for f in sorted(glob.glob('**/*.png', recursive=True)):\n"
                          "    print(f)\n"
                          "    display(Image(filename=f))"))

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
name = f"e89_{LAST}_{FIRST}_HW03_Prob{PROB}.ipynb"
json.dump(nb, open(name, "w", encoding="utf-8"), indent=1)
print("wrote", name)