import os
import shutil

# Folder utama
main_dir = r"c:\Users\diqie\Desktop\Proyek Pengembangan dan Pengoperasian Sistem Machine Learning\muhammad_ivan_LspW-pipeline"
screenshot_dir = os.path.join(main_dir, "screenshot")

# Move screenshots to main folder
files_to_move = [
    "muhammad_ivan_LspW-deployment.png",
    "muhammad_ivan_LspW-monitoring.png",
    "muhammad_ivan_LspW-pylint.png",
    "muhammad_ivan_LspW-grafana-dashboard.png"
]

for f in files_to_move:
    src = os.path.join(screenshot_dir, f)
    dst = os.path.join(main_dir, f)
    
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"✅ Moved: {f}")

# Delete helper scripts
scripts_to_delete = [
    "rename_screenshots.py",
    "serving.py"
]

for s in scripts_to_delete:
    path = os.path.join(main_dir, s)
    if os.path.exists(path):
        os.remove(path)
        print(f"✅ Deleted: {s}")

print("\nCleanup complete!")

# List main folder
print("\nFiles in main folder:")
for f in sorted(os.listdir(main_dir)):
    print(f"  - {f}")
