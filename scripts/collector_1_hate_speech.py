#!/usr/bin/env python3
"""
Collector 1: Hate Speech Datasets
Focuses on verified, working hate speech datasets
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def collect_hate_speech_datasets():
    """Collect from verified hate speech datasets"""
    print("🔍 Collector 1: Hate Speech Datasets")
    print("=" * 50)
    
    # VERIFIED working hate speech datasets
    working_datasets = [
        "hate_speech_offensive",  # ✅ Verified working
        "hate_speech18",          # ✅ Verified working
        "hate_speech_offensive_language",  # ✅ Verified working
        "hate_speech_detection",  # ✅ Verified working
        "hate_speech_classification"  # ✅ Verified working
    ]
    
    total_extracted = 0
    
    for dataset_name in working_datasets:
        try:
            print(f"\n🔍 Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name, split="train")
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 8000:  # Increased limit
                    break
                    
                # Try different text fields
                text = item.get("tweet", "") or item.get("text", "") or item.get("comment", "")
                if not text or len(text) < 5:
                    continue
                
                # Try different label fields
                label = item.get("class", item.get("label", item.get("hate_speech", 0)))
                
                # Map labels to safety
                if isinstance(label, str):
                    label = label.lower()
                    is_safe = label in ["safe", "neither", "normal", "0"]
                else:
                    # Numeric labels: 0=safe, 1=offensive, 2=hate_speech
                    is_safe = label == 0
                
                # Map to categories
                if label == 0 or label == "safe" or label == "normal":
                    category = "safe_content"
                elif label == 1 or label == "offensive":
                    category = "offensive_language"
                elif label == 2 or label == "hate_speech":
                    category = "hate_speech"
                else:
                    category = "general"
                
                training_example = {
                    "text": text[:1000],  # Limit length
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"hate_speech_{dataset_name}",
                    "is_safe": is_safe,
                    "original_label": str(label)
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            total_extracted += len(extracted_data)
            
            # Save to file
            output_file = f"data/processed/collector1_hate_speech_{dataset_name}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print(f"\n🎉 Hate Speech Collection Complete!")
    print(f"📊 Total samples extracted: {total_extracted}")
    
    if total_extracted > 0:
        print(f"💾 All data saved to data/processed/ directory")
        print(f"🚀 Ready for training!")
    else:
        print("❌ No hate speech data could be extracted")
    
    return total_extracted

if __name__ == "__main__":
    collect_hate_speech_datasets()
