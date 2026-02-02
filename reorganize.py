import os
import shutil
import sys
import json
from datetime import datetime

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')

# Create directory structure
dirs = [
    "jd_data/images",
    "jd_data/raw",
    "archive/JD_Details",
    "archive/Old_Assessments",
    "references"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"[OK] Created directory: {d}")

# Move JD data
print("\n[INFO] Moving JD data...")

moved_count = 0

# Move jd/ directory
if os.path.exists("jd"):
    for file in os.listdir("jd"):
        src = os.path.join("jd", file)
        dst = os.path.join("jd_data/images", file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            moved_count += 1
            print(f"  [OK] Copied: {file}")

# Move JD截图/ directory
if os.path.exists("JD截图"):
    for file in os.listdir("JD截图"):
        src = os.path.join("JD截图", file)
        dst = os.path.join("jd_data/images", file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            moved_count += 1
            print(f"  [OK] Copied: {file}")

print(f"\n[OK] Total files copied to jd_data/images/: {moved_count}")

# Move archive files
print("\n[INFO] Moving archive files...")

archive_files = [
    ("43_Positions_Full_JD_Details.md", "archive/JD_Details/"),
    ("44_New_Positions_From_Mobile.md", "archive/JD_Details/"),
    ("AI_Infra_IDE_Positions_70Plus.md", "archive/JD_Details/"),
    ("All_Positions_70Plus_Summary.md", "archive/JD_Details/"),
    ("Comprehensive_Skills_Assessment.md", "archive/Old_Assessments/"),
    ("Mission_Overview.md", "archive/Old_Assessments/"),
]

for file, target_dir in archive_files:
    if os.path.exists(file):
        shutil.move(file, os.path.join(target_dir, file))
        print(f"  [OK] Moved: {file} -> {target_dir}")

# Move reference projects
print("\n[INFO] Moving references...")

if os.path.exists("MODULAR-RAG-MCP-SERVER"):
    shutil.move("MODULAR-RAG-MCP-SERVER", "references/MODULAR-RAG-MCP-SERVER")
    print("  [OK] Moved: MODULAR-RAG-MCP-SERVER")

if os.path.exists("skills"):
    shutil.move("skills", "references/skills")
    print("  [OK] Moved: skills")

# Create metadata.json
print("\n[INFO] Creating metadata.json...")

metadata = {
    "last_updated": datetime.now().strftime("%Y-%m-%d"),
    "total_positions": 87,
    "processed_images": [],
    "last_sync": None,
    "collections": [
        {
            "date": "2026-01-28",
            "source": "招聘网站截图",
            "count": 87,
            "note": "87个年薪70万+AI岗位"
        }
    ]
}

with open("jd_data/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("  [OK] Created: jd_data/metadata.json")

print("\n[SUCCESS] Directory reorganization completed!")
