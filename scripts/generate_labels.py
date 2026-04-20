# auto-generate data/labels.csv from filenames in data/images/
# naming convention: files containing "-edited" are tampered (label=1),
# all others are real (label=0)

from pathlib import Path
import csv

IMAGE_DIR  = Path("data/images")
OUTPUT_CSV = Path("data/labels.csv")

def sort_key(name):
    # sort numerically by the leading number in the filename so that
    # 2, 3, 10, 11 ... rather than 10, 100, 11, 2 (alphabetical order)
    # within the same number, real comes before edited
    stem = Path(name).stem                      # e.g. "15.2-edited" or "15"
    base = stem.split("-")[0].split(".")[0]     # extract leading digits only
    return (int(base), "-edited" in name)

images = sorted(
    (p.name for p in IMAGE_DIR.iterdir()
     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
    key=sort_key
)

rows = [
    {"filename": name, "label": 1 if "-edited" in name else 0}
    for name in images
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
print(f"  real (0):     {sum(r['label'] == 0 for r in rows)}")
print(f"  tampered (1): {sum(r['label'] == 1 for r in rows)}")
