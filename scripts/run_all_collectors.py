#!/usr/bin/env python3
"""
Master Script: Run All Collectors
Runs all individual dataset collectors and combines results
"""

import os
import json
import subprocess
from pathlib import Path

def run_collector(collector_name, script_path):
    """Run a specific collector script"""
    print(f"\n🚀 Running {collector_name}...")
    print("=" * 60)
    
    try:
        result = subprocess.run(["python", script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {collector_name} failed: {e.stderr}")
        return False

def combine_all_collected_data():
    """Combine all collected datasets into one training set"""
    print("\n🔗 Combining All Collected Datasets...")
    print("=" * 60)
    
    # Find all collector JSONL files
    data_dir = Path("data/processed")
    collector_files = list(data_dir.glob("collector*_*.jsonl"))
    
    if not collector_files:
        print("❌ No collector files found to combine")
        return None
    
    print(f"📁 Found {len(collector_files)} collector files:")
    for file in collector_files:
        print(f"   - {file.name}")
    
    # Combine all data
    all_data = []
    file_stats = {}
    
    for file_path in collector_files:
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
        return None
    
    print(f"\n📈 Total samples across all collectors: {len(all_data)}")
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
    train_file = data_dir / "final_combined_train.jsonl"
    val_file = data_dir / "final_combined_validation.jsonl"
    
    print(f"\n💾 Saving final combined dataset...")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Final combined dataset saved:")
    print(f"   - Training: {train_file}")
    print(f"   - Validation: {val_file}")
    
    return len(train_data), len(val_data)

def main():
    """Run all collectors and combine results"""
    print("🚀 Master Dataset Collection Pipeline")
    print("=" * 70)
    print("This will run all individual collectors and combine results")
    print("=" * 70)
    
    # Define all collectors (now including Twitter and Amazon)
    collectors = [
        ("Collector 1: Hate Speech", "scripts/collector_1_hate_speech.py"),
        ("Collector 2: Toxicity", "scripts/collector_2_toxicity.py"),
        ("Collector 3: Reviews", "scripts/collector_3_reviews.py"),
        ("Collector 4: Safety", "scripts/collector_4_safety.py"),
        ("Collector 5: Twitter", "scripts/collector_5_twitter.py"),      # 🆕 NEW!
        ("Collector 6: Amazon", "scripts/collector_6_amazon.py")        # 🆕 NEW!
    ]
    
    # Run all collectors
    collector_results = {}
    
    for collector_name, script_path in collectors:
        success = run_collector(collector_name, script_path)
        collector_results[collector_name] = success
    
    print("\n" + "=" * 70)
    print("📊 Collection Summary:")
    for collector_name, success in collector_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   - {collector_name}: {status}")
    
    # Combine all collected datasets
    successful_collectors = sum(collector_results.values())
    
    if successful_collectors > 0:
        print(f"\n🔗 Combining data from {successful_collectors} successful collectors...")
        result = combine_all_collected_data()
        
        if result:
            train_count, val_count = result
            print(f"\n🎉 Master Pipeline Complete!")
            print(f"🚀 Your final training dataset:")
            print(f"   - Training samples: {train_count}")
            print(f"   - Validation samples: {val_count}")
            print(f"   - Total: {train_count + val_count}")
            print(f"\n💾 Ready for training your Qwen3-4B model!")
            
            # Also combine with existing data if available
            existing_train = Path("data/processed/train.jsonl")
            if existing_train.exists():
                print(f"\n🔗 You also have existing data: {existing_train}")
                print(f"💡 Consider combining both datasets for maximum training data!")
        else:
            print("\n❌ Failed to combine collected datasets")
    else:
        print("\n❌ No collectors were successful")
        print("💡 Check individual collector scripts for debugging")

if __name__ == "__main__":
    main()
