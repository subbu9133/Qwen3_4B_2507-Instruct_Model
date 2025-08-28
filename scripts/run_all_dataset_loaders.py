#!/usr/bin/env python3
"""
Master script to run all dataset loaders and combine results
"""

import os
import json
import glob
from pathlib import Path

def run_twitter_loader():
    """Run Twitter dataset loader"""
    print("🚀 Running Twitter Dataset Loader...")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(["python", "scripts/load_twitter_datasets.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Twitter loader failed: {e.stderr}")
        return False

def run_toxic_loader():
    """Run toxic comments dataset loader"""
    print("\n🚀 Running Toxic Comments Dataset Loader...")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(["python", "scripts/load_toxic_datasets.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Toxic loader failed: {e.stderr}")
        return False

def run_review_loader():
    """Run review dataset loader"""
    print("\n🚀 Running Review Dataset Loader...")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(["python", "scripts/load_review_datasets.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Review loader failed: {e.stderr}")
        return False

def combine_all_datasets():
    """Combine all extracted datasets into one training set"""
    print("\n🔗 Combining All Datasets...")
    print("=" * 50)
    
    # Find all JSONL files
    data_dir = Path("data/processed")
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("❌ No JSONL files found to combine")
        return
    
    print(f"📁 Found {len(jsonl_files)} dataset files:")
    for file in jsonl_files:
        print(f"   - {file.name}")
    
    # Combine all data
    all_data = []
    file_stats = {}
    
    for file_path in jsonl_files:
        print(f"\n📖 Reading {file_path.name}...")
        
        file_data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        file_data.append(item)
            
            print(f"   📊 Loaded {len(file_data)} samples")
            file_stats[file_path.name] = len(file_data)
            all_data.extend(file_data)
            
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")
    
    if not all_data:
        print("❌ No data could be loaded")
        return
    
    print(f"\n📈 Total samples across all datasets: {len(all_data)}")
    print(f"📊 Breakdown by file:")
    for filename, count in file_stats.items():
        print(f"   - {filename}: {count} samples")
    
    # Balance the dataset
    safe_examples = [item for item in all_data if item.get("is_safe", True)]
    unsafe_examples = [item for item in all_data if not item.get("is_safe", True)]
    
    print(f"\n⚖️ Dataset Balance:")
    print(f"   - Safe examples: {len(safe_examples)}")
    print(f"   - Unsafe examples: {len(unsafe_examples)}")
    
    # Balance by undersampling the majority class
    min_count = min(len(safe_examples), len(unsafe_examples))
    
    if len(safe_examples) > min_count:
        import random
        safe_examples = random.sample(safe_examples, min_count)
    
    if len(unsafe_examples) > min_count:
        import random
        unsafe_examples = random.sample(unsafe_examples, min_count)
    
    balanced_data = safe_examples + unsafe_examples
    
    # Shuffle the data
    import random
    random.shuffle(balanced_data)
    
    # Split into train/validation (90/10)
    split_idx = int(len(balanced_data) * 0.9)
    train_data = balanced_data[:split_idx]
    val_data = balanced_data[split_idx:]
    
    print(f"\n🎯 Final Balanced Dataset:")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Save combined dataset
    train_file = data_dir / "combined_train.jsonl"
    val_file = data_dir / "combined_validation.jsonl"
    
    print(f"\n💾 Saving combined dataset...")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Combined dataset saved:")
    print(f"   - Training: {train_file}")
    print(f"   - Validation: {val_file}")
    
    return len(train_data), len(val_data)

def main():
    """Run all dataset loaders and combine results"""
    print("🚀 Master Dataset Loading Pipeline")
    print("=" * 60)
    print("This will run all individual dataset loaders and combine results")
    print("=" * 60)
    
    # Run all loaders
    twitter_success = run_twitter_loader()
    toxic_success = run_toxic_loader()
    review_success = run_review_loader()
    
    print("\n" + "=" * 60)
    print("📊 Dataset Loading Summary:")
    print(f"   - Twitter datasets: {'✅' if twitter_success else '❌'}")
    print(f"   - Toxic comments: {'✅' if toxic_success else '❌'}")
    print(f"   - Review datasets: {'✅' if review_success else '❌'}")
    
    # Combine all datasets
    if any([twitter_success, toxic_success, review_success]):
        print("\n🔗 Combining all extracted datasets...")
        result = combine_all_datasets()
        
        if result:
            train_count, val_count = result
            print(f"\n🎉 Master Pipeline Complete!")
            print(f"🚀 Your final training dataset:")
            print(f"   - Training samples: {train_count}")
            print(f"   - Validation samples: {val_count}")
            print(f"   - Total: {train_count + val_count}")
            print(f"\n💾 Ready for training your Qwen3-4B model!")
        else:
            print("\n❌ Failed to combine datasets")
    else:
        print("\n❌ No datasets were successfully loaded")
        print("💡 Try running individual loaders to debug issues")

if __name__ == "__main__":
    main()
