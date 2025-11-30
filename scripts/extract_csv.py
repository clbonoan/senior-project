# scripts/extract_csv.py
import os, csv, sys, argparse, glob, traceback

''' 
TO RUN THIS SCRIPT:
python scripts/extract_csv.py --images data/images --features texture,lighting,depth --out data/features.csv
'''
# allow imports from shadow feature files (ex: from texture import analyze_texture) from root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from texture import analyze_texture
    from lighting import analyze_lighting
    from depth import analyze_depth
except Exception as e:
    #print("[extract] could not import texture.analyze_texture:", e)
    traceback.print_exc()
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg")

def load_labels(path):
    '''
    csv file with headers (filename,label)
    return: dictionary that maps the key strings -> label (int)
    '''
    # load labels from csv file with columns: filename,label
    lab = {}
    if not path or not os.path.exists(path):
        return lab
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        # assert "filename" in r.fieldnames and "label" in r.fieldnames, \
        #     "labels.csv must have columns: filename,label"
        # for row in r:
        #     v = row["label"]
        #     try: v = int(v)
        #     except: pass
        #     lab[row["filename"]] = v
        # determine which comment is present
        if "filename" in r.fieldnames:
            key_col = "filename"
        elif "image_path" in r.fieldnames or "path" in r.fieldnames:
            key_col = "image_path" if "image_path" in r.fieldnames else "path"
        else:
            raise AssertionError("labels.csv must have column 'filename' or 'image_path', and 'label'")
        assert "label" in r.fieldnames, "labels.csv must have a 'label' column"
        for row in r:
            v = row["label"]
            try:
                v = int(v)
            except:
                pass
            lab[row[key_col]] = v
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

def extract_features_from_image(path, features_to_extract=("texture","lighting","depth"), verbose=False):
    '''
    extract features from all analysis modules
    args:
    - path: path to image file
    - features_to_extract: list of feature types to extract ['texture', 'lighting', 'depth']
    - verbose for detailed error message

    return:
    - combined_features_dict: features only, no rule-based tamper scores
    - errors_list
    '''
    combined_features = {}
    errors = []

    # TEXTURE FEATURES
    if 'texture' in features_to_extract:
        try:
            result = analyze_texture(
                path, 
                visualize=False, 
                compute_tamper_score=False  # DO NOT COMPUTE RULE BASED TAMPER SCORE HERE
            )
            texture_features = result.get("features", {}) or {}

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
            result = analyze_lighting(
                path, 
                show_debug=False,   # lighting.py uses show_debug, not visualize
                compute_tamper_score=False
            )
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
            result = analyze_depth(
                path, 
                visualize=False, 
                compute_tamper_score=False
            )
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
    ap.add_argument("--features", default="texture,lighting,depth", help="Comma-separated list of feature modules to extract")
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
    all_keys = {"filename","label"}     # base columns

    features_to_extract = ['texture', 'lighting', 'depth']

    for i, path in enumerate(files, 1):
        fname = os.path.basename(path)

        print(f"[extract] extracting {i}/{len(files)}: {fname}")
        try:
            # res = analyze_texture(path, visualize=False)   # <- no GUI
            # feats = res.get("features", {})
            feats, errors = extract_features_from_image(
                path,
                features_to_extract=features_to_extract,
                verbose=args.verbose,
            )

            # handle per-module errors with --fail
            if errors:
                if args.verbose:
                    print(f"[{i}/{len(files)}] {fname} feature errors: {errors}")
                if args.fail == "stop":
                    raise RuntimeError("; ".join(errors))
                elif args.fail == "skip":
                    # skip the image
                    continue

        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {fname}: {e}")
            if args.verbose: 
                traceback.print_exc()
            if args.fail == "stop": 
                raise
            elif args.fail == "keep": 
                feats = {}
            else: 
                continue

        # make a row for this image
        row = {"filename": fname}
        if fname in labels: 
            row["label"] = labels[fname]

        # add features to rows
        for k, v in feats.items():
            row[k] = v

        rows.append(row)
        all_keys.update(row.keys())

        print(f"[extract] done {i}/({len(files)})")

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
            clean_row = {}
            # make sure all keys exist and convert values to writable values
            for k in fieldnames:
                v = r.get(k, "")
                if v is None:
                    v = ""
                clean_row[k] = v
            w.writerow(clean_row)

    print(f"[extract] Wrote {len(rows)} rows -> {args.out}")
    print("[extract] done")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
