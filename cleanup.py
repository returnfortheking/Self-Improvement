import shutil
import os

# Remove old directories
dirs_to_remove = ['jd', 'JD截图']
for d in dirs_to_remove:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"[OK] Removed: {d}")

print("[SUCCESS] Cleanup completed")
