import os
import shutil

source_dir = r"phase1 (1)"
target_dir = r"phase1"

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    src_path = os.path.join(source_dir, filename)

    # Skip directories
    if not os.path.isfile(src_path):
        continue

    # Rename pattern: *-0.<ext>  →  *-1.<ext>
    if "-0." in filename:
        new_name = filename.replace("-0.", "-1.")
    else:
        continue  # skip non-matching files

    dst_path = os.path.join(target_dir, new_name)

    # Move with new name
    shutil.move(src_path, dst_path)

print("Done.")
