import os

def audit_datasets(raw_dir="datasets/raw"):
    stats = {
        "total_files": 0,
        "empty_files": 0,
        "extensions": {},
        "categories": {}
    }
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return stats
        
    for root, dirs, files in os.walk(raw_dir):
        category = os.path.basename(root)
        if category == "raw":
            continue
            
        stats["categories"][category] = len(files)
        
        for file in files:
            stats["total_files"] += 1
            ext = os.path.splitext(file)[1].lower()
            stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
            
            path = os.path.join(root, file)
            if os.path.getsize(path) == 0:
                stats["empty_files"] += 1
                
    print("="*50)
    print("DATASET AUDIT REPORT")
    print("="*50)
    print(f"Total Files: {stats['total_files']}")
    print(f"Empty/Corrupted Files: {stats['empty_files']}")
    print("\nFile Types:")
    for ext, count in stats["extensions"].items():
        print(f"  {ext}: {count}")
    print("\nCategories:")
    for cat, count in stats["categories"].items():
        print(f"  {cat}: {count}")
    print("="*50)
    
if __name__ == "__main__":
    audit_datasets()
