#!/usr/bin/env python3
"""
Collector 6: Amazon Datasets
Directly loads and processes Amazon review data
"""

import json
import os
import sys
from pathlib import Path

def collect_amazon_datasets():
    """Collect Amazon data directly from Hugging Face"""
    print("🔍 Collector 6: Amazon Datasets")
    print("=" * 50)
    print("Directly loading Amazon datasets from Hugging Face")
    print("=" * 50)
    
    try:
        # Try to load Amazon datasets from Hugging Face
        print("\n🔄 Trying to load Amazon datasets from Hugging Face...")
        
        try:
            from datasets import load_dataset
            
            # Try different Amazon datasets that are known to work
            amazon_datasets = [
                "amazon_polarity",
                "amazon_us_reviews",
                "amazon_customer_reviews"
            ]
            
            for dataset_name in amazon_datasets:
                try:
                    print(f"\n🔍 Trying: {dataset_name}")
                    dataset = load_dataset(dataset_name, split="train")
                    print(f"✅ Successfully loaded: {dataset_name}")
                    
                    # Extract data directly
                    extracted_data = []
                    max_samples = 8000
                    
                    for i, item in enumerate(dataset):
                        if len(extracted_data) >= max_samples:
                            break
                        
                        # Get review text from various possible fields
                        review_text = item.get("review_body", "") or item.get("text", "") or item.get("content", "") or item.get("review", "")
                        if not review_text or len(review_text) < 10:
                            continue
                        
                        # Get rating/sentiment
                        rating = item.get("star_rating", item.get("rating", item.get("score", item.get("label", 5))))
                        try:
                            rating = float(rating)
                        except:
                            rating = 5.0
                        
                        # Determine safety based on rating and content
                        safety_keywords = [
                            "harmful", "dangerous", "inappropriate", "offensive", "toxic",
                            "hate", "violence", "discrimination", "harassment", "bullying",
                            "awful", "terrible", "horrible", "disgusting", "vile", "bad",
                            "worst", "kill", "die", "stupid", "idiot", "moron",
                            "useless", "garbage", "trash", "broken", "defective", "fake"
                        ]
                        
                        text_lower = review_text.lower()
                        contains_safety_content = any(keyword in text_lower for keyword in safety_keywords)
                        
                        # Determine if it's safe
                        if contains_safety_content:
                            is_safe = False
                            category = "amazon_review_safety"
                        elif rating <= 2.0:  # Low rating might indicate safety issues
                            is_safe = False
                            category = "amazon_low_rating"
                        else:
                            is_safe = True
                            category = "amazon_positive_review"
                        
                        training_example = {
                            "text": review_text[:800],  # Limit length
                            "category": category,
                            "label": "SAFE" if is_safe else "UNSAFE",
                            "source": f"amazon_{dataset_name}",
                            "is_safe": is_safe,
                            "rating": rating,
                            "contains_safety_keywords": contains_safety_content,
                            "platform": "amazon"
                        }
                        extracted_data.append(training_example)
                    
                    print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
                    
                    if len(extracted_data) > 0:
                        # Save to file
                        output_file = f"data/processed/collector6_amazon_{dataset_name}.jsonl"
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        with open(output_file, "w", encoding="utf-8") as f:
                            for item in extracted_data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
                        print(f"💾 Saved to: {output_file}")
                        
                        print(f"\n🎉 Amazon Collection Complete!")
                        print(f"📊 Total samples extracted: {len(extracted_data)}")
                        return len(extracted_data)
                    
                except Exception as e:
                    print(f"❌ Failed to load {dataset_name}: {e}")
                    continue
            
            # If no Hugging Face datasets worked, show instructions
            print("\n❌ No Hugging Face Amazon datasets could be loaded")
            print("💡 Please check dataset availability or try alternative sources")
            
        except ImportError:
            print("❌ Hugging Face datasets library not available")
            print("💡 Install with: pip install datasets")
        
        return 0
            
    except Exception as e:
        print(f"❌ Error in Amazon collection: {e}")
        return 0

if __name__ == "__main__":
    collect_amazon_datasets()
