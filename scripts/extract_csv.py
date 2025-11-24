# scripts/extract_csv.py
import os, csv, sys, argparse, glob, traceback

# allow "from texture import analyze_image" from root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from texture import analyze_texture
    from lighting import analyze_lighting
    from depth import analyze_depth
except Exception as e:
    #print("[extract] Could not import texture.analyze_image:", e)
    traceback.print_exc()
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg")

def load_labels(path):
    # load labels from csv file with columns: filename, label
    lab = {}
    if not path or not os.path.exists(path):
        return lab
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        assert "filename" in r.fieldnames and "label" in r.fieldnames, \
            "labels.csv must have columns: filename,label"
        for row in r:
            v = row["label"]
            try: v = int(v)
            except: pass
            lab[row["filename"]] = v
    return lab

def list_images(root):
    # find all images in directory or return a single file
    if os.path.isdir(root):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        return sorted(files)
    elif os.path.isfile(root) and root.lower().endswith(IMG_EXTS):
        return [root]
    return []

def extract_features_from_image(path, features_to_extract, verbose=False):
    '''
    extract features from all analysis modules
    args:
    - path: path to image file
    - features_to_extract: list of feature types to extract ['texture', 'lighting', 'depth']
    - verbose for detailed error message

    return:
    - dict: combined features from all modules (excludes rule-based tampered scores)
    '''
    combined_features = {}
    errors = []

    # TEXTURE FEATURES
    if 'texture' in features_to_extract:
        try:
            result = analyze_texture(path, visualize=False, compute_tamper_score=False)
            texture_features = result.get("features", {})

            # prefix texture features to avoid name collisions
            for key, val in texture_features.items():
                combined_features[f"texture_{key}"] = val
        except Exception as e:
            errors.append(f"texture: {e}")
            if verbose:
                print(f"  [texture] Error: {e}")
                traceback.print_exc()
    
    # LIGHTING FEATURES
    if 'lighting' in features_to_extract:
        try:
            result = analyze_lighting(path, visualize=False, compute_tamper_score=False)
            lighting_features = result.get("features", {})

            # prefix lighting features
            for key, val in lighting_features.items():
                combined_features[f"lighting_{key}"] = val
        except Exception as e:
            errors.append(f"lighting: {e}")
            if verbose:
                print(f"  [lighting] Error: {e}")
                traceback.print_exc()
    
    # DEPTH FEATURES
    if 'depth' in features_to_extract:
        try:
            result = analyze_depth(path, visualize=False, compute_tamper_score=False)
            depth_features = result.get("features", {})

            # prefix lighting features
            for key, val in depth_features.items():
                combined_features[f"depth_{key}"] = val
        except Exception as e:
            errors.append(f"depth: {e}")
            if verbose:
                print(f"  [depth] Error: {e}")
                traceback.print_exc()   

    return combined_features, errors     

def main():
    ap = argparse.ArgumentParser(description="Extract features to CSV for ML training")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--fail", default="skip", choices=["skip","keep","stop"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[extract] start")
    print(f"[extract] images={args.images} labels={args.labels} out={args.out}")

    labels = load_labels(args.labels)
    files = list_images(args.images)
    print(f"[extract] found {len(files)} image(s)")

    if not files:
        print(f"[extract] No images found under: {args.images}  (expected {IMG_EXTS})")
        sys.exit(1)

    rows = []
    all_keys = {"filename","label"}

    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)
        try:
            res = analyze_texture(path, visualize=False)   # <- no GUI
            feats = res.get("features", {})
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {fname}: {e}")
            if args.verbose: traceback.print_exc()
            if args.fail == "stop": raise
            elif args.fail == "keep": feats = {}
            else: continue

        row = {"filename": fname, **feats}
        if fname in labels: row["label"] = labels[fname]
        rows.append(row)
        all_keys.update(row.keys())

        if args.verbose and (i % 5 == 0 or i == len(files)):
            print(f"[extract] Processed {i}/{len(files)}")

    if not rows:
        print("[extract] No rows generated; exiting.")
        sys.exit(2)

    fieldnames = sorted(all_keys)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for k in fieldnames:
                if k not in r: r[k] = ""
            w.writerow(r)

    print(f"[extract] Wrote {len(rows)} rows -> {args.out}")
    print("[extract] done")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
