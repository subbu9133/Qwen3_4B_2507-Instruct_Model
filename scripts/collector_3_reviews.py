#!/usr/bin/env python3
"""
Collector 3: Review Datasets
Focuses on verified, working review and sentiment datasets
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def collect_review_datasets():
    """Collect from verified review datasets"""
    print("🔍 Collector 3: Review Datasets")
    print("=" * 50)
    
    # VERIFIED working review datasets
    working_datasets = [
        "yelp_review_full",  # ✅ Verified working - Yelp Reviews
        "yelp_review_polarity",  # ✅ Verified working - Yelp Polarity
        "imdb",  # ✅ Verified working - IMDB Reviews
        "rotten_tomatoes",  # ✅ Verified working - Rotten Tomatoes
        "sst2",  # ✅ Verified working - Stanford Sentiment Treebank
        "ag_news"  # ✅ Verified working - AG News
    ]
    
    total_extracted = 0
    
    for dataset_name in working_datasets:
        try:
            print(f"\n🔍 Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name, split="train")
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            safety_keywords = [
                "harmful", "dangerous", "inappropriate", "offensive", "toxic",
                "hate", "violence", "discrimination", "harassment", "bullying",
                "awful", "terrible", "horrible", "disgusting", "vile", "bad",
                "worst", "hate", "kill", "die", "stupid", "idiot", "moron"
            ]
            
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 5000:  # Limit for reviews
                    break
                    
                # Try different text fields
                text = item.get("text", "") or item.get("review", "") or item.get("sentence", "")
                if not text or len(text) < 10:
                    continue
                
                # Check for safety-related content
                text_lower = text.lower()
                contains_safety_content = any(keyword in text_lower for keyword in safety_keywords)
                
                # Also check sentiment for some datasets
                sentiment = item.get("label", item.get("sentiment", 0))
                
                # Determine if it's safe based on content and sentiment
                if contains_safety_content:
                    is_safe = False
                    category = "review_safety"
                elif sentiment in [0, 1, "negative", "neg"]:  # Negative sentiment
                    is_safe = False
                    category = "negative_review"
                else:
                    is_safe = True
                    category = "positive_review"
                
                training_example = {
                    "text": text[:800],  # Limit length
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"reviews_{dataset_name}",
                    "is_safe": is_safe,
                    "sentiment": str(sentiment),
                    "contains_safety_keywords": contains_safety_content
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            total_extracted += len(extracted_data)
            
            # Save to file
            output_file = f"data/processed/collector3_reviews_{dataset_name}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print(f"\n🎉 Review Collection Complete!")
    print(f"📊 Total samples extracted: {total_extracted}")
    
    if total_extracted > 0:
        print(f"💾 All data saved to data/processed/ directory")
        print(f"🚀 Ready for training!")
    else:
        print("❌ No review data could be extracted")
    
    return total_extracted

if __name__ == "__main__":
    collect_review_datasets()
