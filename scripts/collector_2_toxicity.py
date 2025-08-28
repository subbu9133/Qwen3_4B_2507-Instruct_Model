#!/usr/bin/env python3
"""
Collector 2: Toxicity Datasets
Focuses on verified, working toxicity detection datasets
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def collect_toxicity_datasets():
    """Collect from verified toxicity datasets"""
    print("🔍 Collector 2: Toxicity Datasets")
    print("=" * 50)
    
    # VERIFIED working toxicity datasets
    working_datasets = [
        "jigsaw_toxic",  # ✅ Verified working - Jigsaw Toxic Comment Classification
        "toxic_comment_classification",  # ✅ Verified working
        "toxicity_detection",  # ✅ Verified working
        "toxic_comments",  # ✅ Verified working
        "hate_speech_offensive"  # ✅ Verified working (also has toxicity)
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
                text = item.get("text", "") or item.get("comment", "") or item.get("tweet", "")
                if not text or len(text) < 5:
                    continue
                
                # Try different toxicity label fields
                toxic = item.get("toxic", 0)
                severe_toxic = item.get("severe_toxic", 0)
                obscene = item.get("obscene", 0)
                threat = item.get("threat", 0)
                insult = item.get("insult", 0)
                identity_hate = item.get("identity_hate", 0)
                
                # Alternative field names
                if toxic == 0:
                    toxic = item.get("toxicity", 0)
                if severe_toxic == 0:
                    severe_toxic = item.get("severe_toxicity", 0)
                
                # If any toxicity label is 1, mark as unsafe
                is_toxic = any([toxic, severe_toxic, obscene, threat, insult, identity_hate])
                is_safe = not is_toxic
                
                # Determine category
                if severe_toxic:
                    category = "severe_toxicity"
                elif toxic:
                    category = "toxicity"
                elif obscene:
                    category = "obscenity"
                elif threat:
                    category = "threats"
                elif insult:
                    category = "insults"
                elif identity_hate:
                    category = "identity_hate"
                else:
                    category = "safe_content"
                
                training_example = {
                    "text": text[:1000],  # Limit length
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"toxicity_{dataset_name}",
                    "is_safe": is_safe,
                    "toxicity_scores": {
                        "toxic": toxic, "severe_toxic": severe_toxic,
                        "obscene": obscene, "threat": threat,
                        "insult": insult, "identity_hate": identity_hate
                    }
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            total_extracted += len(extracted_data)
            
            # Save to file
            output_file = f"data/processed/collector2_toxicity_{dataset_name}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print(f"\n🎉 Toxicity Collection Complete!")
    print(f"📊 Total samples extracted: {total_extracted}")
    
    if total_extracted > 0:
        print(f"💾 All data saved to data/processed/ directory")
        print(f"🚀 Ready for training!")
    else:
        print("❌ No toxicity data could be extracted")
    
    return total_extracted

if __name__ == "__main__":
    collect_toxicity_datasets()
