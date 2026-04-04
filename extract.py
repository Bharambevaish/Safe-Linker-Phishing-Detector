import json

with open("safelinkerfinal.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("safelinker_extracted.py", "w", encoding="utf-8") as f:
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            f.write(source + "\n\n")
