import sys
import traceback
from pathlib import Path
import pandas as pd

# FIND PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# allow imports from project root
sys.path.append(str(PROJECT_ROOT))

# IMPORT MODULES
try:
    from texture import analyze_texture
    from lighting import analyze_lighting
    from depth import analyze_depth
except Exception as e:
    #print("[extract] could not import texture.analyze_texture:", e)
    traceback.print_exc()
    sys.exit(1)

# DATA PATHS - choose dataset location
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"

TEXTURE_OUT = DATA_DIR / "texture_features.csv"
LIGHTING_OUT = DATA_DIR / "lighting_features.csv"
DEPTH_OUT = DATA_DIR / "depth_features.csv"

# LOAD LABELS.CSV
def load_labels(labels_csv_path):
    df = pd.read_csv(labels_csv_path)

    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "labels.csv must contain columns: filename, label"
        )
    
    df = df.rename(columns={"filename": "image_id"})

    # ensure correct types
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    return df

# MODULE RUNNING FUNCTIONS
def run_texture(image_path):
    return analyze_texture(
        str(image_path),
        visualize=False,
        compute_tamper_score=True
    )

def run_lighting(image_path):
    return analyze_lighting(
        str(image_path),
        show_debug=False,
        compute_tamper_score=True
    )

def run_depth(image_path):
    return analyze_depth(
        str(image_path),
        visualize=False,
        compute_tamper_score=True,
        sample_step=4,
        min_shadow_area=300,
        min_perimeter=30
    )

# SAFE RUNNER
def safe_analyze(module_name, analyze_function, image_path):
    '''
    run one module safely; if one image fails, return safe values 
    instead of crashing the whole script
    '''
    try:
        result = analyze_function(image_path)

        if not isinstance(result, dict):
            raise ValueError(f"{module_name} did not return a dict")

        features = result.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"{module_name} returned invalid 'features'")

        out = {
            "features": features,
            "tamper_score": result.get("tamper_score", None),
            "error": None,
        }
        return out

    except Exception as e:
        print(f"[ERROR] {module_name} failed on {image_path.name}: {e}")
        traceback.print_exc()
        return {
            "features": {},
            "tamper_score": None,
            "error": str(e),
        }
    
# BUILD ROWS
def build_feature_rows(labels_df):
    texture_rows = []
    lighting_rows = []
    depth_rows = []

    total = len(labels_df)

    for idx, row in labels_df.iterrows():
        image_id = row["image_id"]
        label = row["label"]
        image_path = IMAGES_DIR / image_id

        print(f"[{idx + 1}/{total}] Processing {image_id}")

        if not image_path.exists():
            print(f"[WARNING] Image not found {image_path}")
            continue

        # Run each module safely
        tex = safe_analyze("texture", run_texture, image_path)
        light = safe_analyze("lighting", run_lighting, image_path)
        dep = safe_analyze("depth", run_depth, image_path)

        texture_row = {
            "image_id": image_id,
            "label": label,
            "rule_score_texture": tex["tamper_score"],
            "texture_error": tex["error"],
            **tex["features"],
        }
        
        print("DEBUG texture tamper score:", tex["tamper_score"])

        # tex = pd.read_csv("data/texture_features.csv")
        # print(tex.groupby("label").mean(numeric_only=True).T)

        lighting_row = {
            "image_id": image_id,
            "label": label,
            "rule_score_lighting": light["tamper_score"],
            "lighting_error": light["error"],
            **light["features"],
        }

        depth_row = {
            "image_id": image_id,
            "label": label,
            "rule_score_depth": dep["tamper_score"],
            "depth_error": dep["error"],
            **dep["features"],
        }
        
        texture_rows.append(texture_row)
        lighting_rows.append(lighting_row)
        depth_rows.append(depth_row)

    return texture_rows, lighting_rows, depth_rows

# SAVE CSV FILES
def save_rows(rows, output_path):
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[WARNING] No rows to save for {output_path.name}")
        return df
    
    # put image_id and label first to read easier
    front_cols = [c for c in ["image_id", "label"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    return df

# MAIN
if __name__ == "__main__":
    print("Loading labels...")
    labels_df = load_labels(LABELS_CSV)

    print(f"Found {len(labels_df)} labeled images")
    print("Building features CSV files...")

    texture_rows, lighting_rows, depth_rows = build_feature_rows(labels_df)

    print("\nSaving CSV files...")
    texture_df = save_rows(texture_rows, TEXTURE_OUT)
    lighting_df = save_rows(lighting_rows, LIGHTING_OUT)
    depth_df = save_rows(depth_rows, DEPTH_OUT)

    print("\nCOMPLETE")
    print(f"Texture rows: {len(texture_df)}")
    print(f"Lighting rows: {len(lighting_df)}")
    print(f"Depth rows: {len(depth_df)}")
