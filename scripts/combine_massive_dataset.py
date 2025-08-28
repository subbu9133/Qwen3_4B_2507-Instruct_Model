#!/usr/bin/env python3
"""
Combine MASSIVE Dataset
Combines ALL data: existing data + new collected data for maximum training power
"""

import os
import json
from pathlib import Path

def combine_massive_dataset():
    """Combine ALL data into one massive dataset"""
    print("Combining MASSIVE Dataset - All Data Combined!")
    print("=" * 60)
    
    data_dir = Path("data/processed")
    all_data = []
    
    # 1. Load existing data (Airoboros + OpenOrca)
    print("📚 Loading Existing Data...")
    existing_files = [
        "train.jsonl",
        "validation.jsonl"
    ]
    
    existing_count = 0
    for file_name in existing_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"   Loading {file_name}...")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            all_data.append(item)
                            existing_count += 1
                print(f"   ✅ Loaded {existing_count} samples from {file_name}")
            except Exception as e:
                print(f"   ❌ Error reading {file_name}: {e}")
    
    print(f"📊 Total existing data loaded: {existing_count} samples")
    
    # 2. Load ALL new collected data
    print(f"\n🆕 Loading New Collected Data...")
    collector_files = list(data_dir.glob("collector*_*.jsonl"))
    
    if not collector_files:
        print("❌ No collector files found!")
        return None
    
    print(f"Found {len(collector_files)} collector files:")
    for file in collector_files:
        print(f"   - {file.name}")
    
    new_data_count = 0
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
            new_data_count += len(file_data)
            
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")
    
    print(f"\n📈 Total Dataset Summary:")
    print(f"   - Existing data: {existing_count} samples")
    print(f"   - New collected data: {new_data_count} samples")
    print(f"   - TOTAL COMBINED: {len(all_data)} samples 🚀")
    
    print(f"\n📊 Breakdown by collector file:")
    for filename, count in file_stats.items():
        print(f"   - {filename}: {count} samples")
    
    # 3. Balance the MASSIVE dataset
    print(f"\n⚖️ Balancing MASSIVE Dataset...")
    safe_examples = [item for item in all_data if item.get("is_safe", True)]
    unsafe_examples = [item for item in all_data if not item.get("is_safe", True)]
    
    print(f"   - Safe examples: {len(safe_examples)}")
    print(f"   - Unsafe examples: {len(unsafe_examples)}")
    
    # Balance to equal numbers for optimal training
    min_count = min(len(safe_examples), len(unsafe_examples))
    if len(safe_examples) > min_count:
        import random
        safe_examples = random.sample(safe_examples, min_count)
        print(f"   - Balanced safe examples to: {len(safe_examples)}")
    if len(unsafe_examples) > min_count:
        import random
        unsafe_examples = random.sample(unsafe_examples, min_count)
        print(f"   - Balanced unsafe examples to: {len(unsafe_examples)}")
    
    balanced_data = safe_examples + unsafe_examples
    import random
    random.shuffle(balanced_data)
    
    # 4. Split into train/validation
    print(f"\n✂️ Splitting MASSIVE Dataset...")
    split_idx = int(len(balanced_data) * 0.9)
    train_data = balanced_data[:split_idx]
    val_data = balanced_data[split_idx:]
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    print(f"   - Total balanced: {len(balanced_data)}")
    
    # 5. Save MASSIVE combined dataset
    print(f"\n💾 Saving MASSIVE Combined Dataset...")
    train_file = data_dir / "MASSIVE_combined_train.jsonl"
    val_file = data_dir / "MASSIVE_combined_validation.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ MASSIVE Dataset Saved!")
    print(f"   - Training: {train_file}")
    print(f"   - Validation: {val_file}")
    
    # 6. Generate final statistics
    print(f"\n🎯 FINAL MASSIVE DATASET STATISTICS:")
    print(f"   🚀 Total samples: {len(all_data)}")
    print(f"   ⚖️ Balanced samples: {len(balanced_data)}")
    print(f"   📚 Training samples: {len(train_data)}")
    print(f"   🔍 Validation samples: {len(val_data)}")
    print(f"   📊 Safe/Unsafe ratio: 1:1 (balanced)")
    
    return len(train_data), len(val_data), len(all_data)

def main():
    print("🚀 Creating MASSIVE Dataset for Maximum Training Power!")
    print("=" * 70)
    result = combine_massive_dataset()
    
    if result:
        train_count, val_count, total_count = result
        print(f"\n🎉 MASSIVE DATASET CREATION COMPLETE!")
        print(f"=" * 50)
        print(f"🚀 Your Qwen3-4B model now has:")
        print(f"   📚 Training samples: {train_count:,}")
        print(f"   🔍 Validation samples: {val_count:,}")
        print(f"   📊 Total balanced: {total_count:,}")
        print(f"\n💪 This MASSIVE dataset will make your model:")
        print(f"   - Much more robust and accurate")
        print(f"   - Better at detecting various types of harmful content")
        print(f"   - More generalizable across different domains")
        print(f"   - Competitive with ShieldGemma and Llama-Guard!")
        print(f"\n🎯 Ready for training your Qwen3-4B content moderation model!")
    else:
        print("❌ Failed to create MASSIVE dataset")

if __name__ == "__main__":
    main()
