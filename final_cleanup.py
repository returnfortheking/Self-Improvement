import shutil
import os

# Remove old directories
dirs_to_remove = ['jd', 'JD截图', 'skills']
removed = []

for d in dirs_to_remove:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            removed.append(d)
            print(f"[OK] Removed: {d}")
        except Exception as e:
            print(f"[ERROR] Failed to remove {d}: {e}")
    else:
        print(f"[SKIP] Not found: {d}")

print(f"\n[SUCCESS] Cleanup completed!")
print(f"Removed: {', '.join(removed)}")
