#!/usr/bin/env python3
"""
Combine Existing Collected Data
Combines all the data files that were already collected by individual collectors
"""

import os
import json
from pathlib import Path

def combine_existing_data():
    """Combine all existing collected data files"""
    print("Combining Existing Collected Data")
    print("=" * 50)
    
    data_dir = Path("data/processed")
    collector_files = list(data_dir.glob("collector*_*.jsonl"))
    
    if not collector_files:
        print("No collector files found to combine")
        return None
    
    print(f"Found {len(collector_files)} collector files:")
    for file in collector_files:
        print(f"   - {file.name}")
    
    all_data = []
    file_stats = {}
    
    for file_path in collector_files:
        print(f"\nReading {file_path.name}...")
        file_data = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        file_data.append(item)
            
            print(f"   Loaded {len(file_data)} samples")
            file_stats[file_path.name] = len(file_data)
            all_data.extend(file_data)
            
        except Exception as e:
            print(f"   Error reading {file_path.name}: {e}")
    
    if not all_data:
        print("No data could be loaded")
        return None
    
    print(f"\nTotal samples across all collectors: {len(all_data)}")
    print(f"Breakdown by file:")
    for filename, count in file_stats.items():
        print(f"   - {filename}: {count} samples")
    
    # Balance the dataset
    safe_examples = [item for item in all_data if item.get("is_safe", True)]
    unsafe_examples = [item for item in all_data if not item.get("is_safe", True)]
    
    print(f"\nDataset Balance:")
    print(f"   - Safe examples: {len(safe_examples)}")
    print(f"   - Unsafe examples: {len(unsafe_examples)}")
    
    # Balance to equal numbers
    min_count = min(len(safe_examples), len(unsafe_examples))
    if len(safe_examples) > min_count:
        import random
        safe_examples = random.sample(safe_examples, min_count)
    if len(unsafe_examples) > min_count:
        import random
        unsafe_examples = random.sample(unsafe_examples, min_count)
    
    balanced_data = safe_examples + unsafe_examples
    import random
    random.shuffle(balanced_data)
    
    # Split into train/validation
    split_idx = int(len(balanced_data) * 0.9)
    train_data = balanced_data[:split_idx]
    val_data = balanced_data[split_idx:]
    
    print(f"\nFinal Balanced Dataset:")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Save combined dataset
    train_file = data_dir / "final_combined_train.jsonl"
    val_file = data_dir / "final_combined_validation.jsonl"
    
    print(f"\nSaving final combined dataset...")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Final combined dataset saved:")
    print(f"   - Training: {train_file}")
    print(f"   - Validation: {val_file}")
    
    return len(train_data), len(val_data)

def main():
    print("Combining all existing collected data...")
    result = combine_existing_data()
    
    if result:
        train_count, val_count = result
        print(f"\nCombination Complete!")
        print(f"Final training dataset:")
        print(f"   - Training samples: {train_count}")
        print(f"   - Validation samples: {val_count}")
        print(f"   - Total: {train_count + val_count}")
        print(f"\nReady for training your Qwen3-4B model!")
    else:
        print("Failed to combine data")

if __name__ == "__main__":
    main()
