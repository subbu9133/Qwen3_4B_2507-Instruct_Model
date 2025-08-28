#!/usr/bin/env python3
"""
Load review datasets for content moderation training
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def load_amazon_reviews():
    """Try multiple Amazon review datasets"""
    print("🔍 Loading Amazon Review Datasets...")
    
    # List of possible Amazon review datasets
    dataset_options = [
        "amazon_us_reviews",
        "amazon_reviews",
        "amazon_books",
        "amazon_products",
        "amazon_customer_reviews"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            
            # Try different configurations
            configs = ["Books_v1_02", "Books_v1_01", "Electronics_v1_00", None]
            
            for config in configs:
                try:
                    if config:
                        dataset = datasets.load_dataset(dataset_name, config, split="train")
                    else:
                        dataset = datasets.load_dataset(dataset_name, split="train")
                    
                    print(f"✅ Successfully loaded: {dataset_name} with config: {config}")
                    
                    # Extract data
                    extracted_data = []
                    safety_keywords = [
                        "harmful", "dangerous", "inappropriate", "offensive", "toxic",
                        "hate", "violence", "discrimination", "harassment", "bullying",
                        "awful", "terrible", "horrible", "disgusting", "vile"
                    ]
                    
                    for item in tqdm(dataset, desc=f"Processing {dataset_name} {config}"):
                        if len(extracted_data) >= 5000:  # Limit for testing
                            break
                            
                        review_text = item.get("review_body", "") or item.get("text", "") or item.get("review", "")
                        if not review_text or len(review_text) < 10:
                            continue
                        
                        # Check for safety-related content
                        contains_safety_content = any(keyword in review_text.lower() for keyword in safety_keywords)
                        
                        if contains_safety_content:
                            # Mark as potentially unsafe
                            is_safe = False
                            category = "product_review_safety"
                            
                            training_example = {
                                "text": review_text[:500],  # Limit length
                                "category": category,
                                "label": "SAFE" if is_safe else "UNSAFE",
                                "source": f"amazon_reviews_{dataset_name}_{config}",
                                "is_safe": is_safe,
                                "rating": item.get("star_rating", item.get("rating", 0))
                            }
                            extracted_data.append(training_example)
                    
                    print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name} {config}")
                    
                    if len(extracted_data) > 0:
                        # Save to file
                        output_file = f"data/processed/amazon_reviews_{dataset_name}_{config}.jsonl"
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        with open(output_file, "w", encoding="utf-8") as f:
                            for item in extracted_data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
                        print(f"💾 Saved to: {output_file}")
                        return extracted_data
                    
                except Exception as e:
                    print(f"❌ Failed config {config}: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No Amazon review datasets could be loaded")
    return []

def load_other_review_datasets():
    """Try other review and comment datasets"""
    print("🔍 Loading Other Review and Comment Datasets...")
    
    # List of other review datasets
    dataset_options = [
        "yelp_review_full",
        "yelp_review_polarity",
        "imdb",
        "rotten_tomatoes",
        "sst2",
        "ag_news"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name, split="train")
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            safety_keywords = [
                "harmful", "dangerous", "inappropriate", "offensive", "toxic",
                "hate", "violence", "discrimination", "harassment", "bullying"
            ]
            
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 3000:  # Limit for testing
                    break
                    
                # Try different text fields
                text = item.get("text", "") or item.get("review", "") or item.get("sentence", "")
                if not text:
                    continue
                
                # Check for safety-related content
                contains_safety_content = any(keyword in text.lower() for keyword in safety_keywords)
                
                if contains_safety_content:
                    # Mark as potentially unsafe
                    is_safe = False
                    category = "review_safety"
                    
                    training_example = {
                        "text": text[:500],  # Limit length
                        "category": category,
                        "label": "SAFE" if is_safe else "UNSAFE",
                        "source": f"other_reviews_{dataset_name}",
                        "is_safe": is_safe
                    }
                    extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            
            if len(extracted_data) > 0:
                # Save to file
                output_file = f"data/processed/other_reviews_{dataset_name}.jsonl"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in extracted_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                print(f"💾 Saved to: {output_file}")
                return extracted_data
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No other review datasets could be loaded")
    return []

def main():
    """Load all review datasets"""
    print("🚀 Loading Review Datasets for Content Moderation")
    print("=" * 50)
    
    # Load Amazon reviews data
    amazon_data = load_amazon_reviews()
    
    print("\n" + "=" * 50)
    
    # Load other review data
    other_reviews_data = load_other_review_datasets()
    
    print("\n" + "=" * 50)
    
    # Summary
    total_samples = len(amazon_data) + len(other_reviews_data)
    print(f"🎉 Review Data Loading Complete!")
    print(f"📊 Total samples extracted: {total_samples}")
    print(f"   - Amazon Reviews: {len(amazon_data)}")
    print(f"   - Other Reviews: {len(other_reviews_data)}")
    
    if total_samples > 0:
        print(f"💾 Data saved to data/processed/ directory")
        print(f"🚀 You can now combine this with your existing data!")

if __name__ == "__main__":
    main()
