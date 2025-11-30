# Detecting Image Tampering Through Shadow Features in Urban Infrastructure
This is our Fall 2025 - Spring 2026 senior project as a requirement of CSUB's Computer Science degree curriculum.

### Run (to extract features from shadow feature modules (texture, lighting, depth)):
python scripts/extract_csv.py --images data/images --labels data/labels.csv --features texture,lighting,depth --out data/features.csv

>[!NOTE]
> RUNNING THE ABOVE DOES THE FOLLOWING: reads all images from data/images/, uses data/labels.csv for the ground truth labels, saves the extracted measurements to data/features.csv

> [!NOTE]
> Place your images in data/images/ (this is not tracked in Git)

