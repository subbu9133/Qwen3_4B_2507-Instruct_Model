#!/usr/bin/env python3
"""
Load Twitter datasets for content moderation training
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def load_twitter_hate_speech():
    """Try multiple Twitter hate speech datasets"""
    print("🔍 Loading Twitter Hate Speech Datasets...")
    
    # List of possible Twitter hate speech datasets
    dataset_options = [
        "t-davidson/hate_speech_and_offensive_language",
        "hate_speech_offensive",
        "hate_speech",
        "offensive_language",
        "t-davidson/hate_speech",
        "t-davidson/offensive_language"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name)
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            for item in tqdm(dataset["train"], desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 5000:  # Limit for testing
                    break
                    
                text = item.get("tweet", "") or item.get("text", "")
                if not text:
                    continue
                
                # Try different label fields
                label = item.get("class", item.get("label", item.get("hate_speech", 0)))
                is_safe = label == 2 or label == "neither" or label == "safe"
                
                # Map to categories
                if label == 0 or label == "hate_speech":
                    category = "hate_speech"
                elif label == 1 or label == "offensive":
                    category = "offensive_language"
                else:
                    category = "general"
                
                training_example = {
                    "text": text,
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"twitter_{dataset_name.replace('/', '_')}",
                    "is_safe": is_safe
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            
            # Save to file
            output_file = f"data/processed/twitter_hate_speech_{dataset_name.replace('/', '_')}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            return extracted_data
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No Twitter hate speech datasets could be loaded")
    return []

def load_twitter_cyberbullying():
    """Try multiple Twitter cyberbullying datasets"""
    print("🔍 Loading Twitter Cyberbullying Datasets...")
    
    # List of possible cyberbullying datasets
    dataset_options = [
        "cyberbullying_detection/cyberbullying_tweets",
        "cyberbullying_tweets",
        "bullying_detection",
        "cyberbullying",
        "bullying_tweets"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name)
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            for item in tqdm(dataset["train"], desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 5000:  # Limit for testing
                    break
                    
                text = item.get("tweet_text", "") or item.get("text", "") or item.get("tweet", "")
                if not text:
                    continue
                
                # Check for cyberbullying indicators
                is_cyberbullying = False
                for key in item.keys():
                    if "bully" in key.lower() or "harass" in key.lower():
                        if item[key] == 1 or item[key] == True:
                            is_cyberbullying = True
                            break
                
                is_safe = not is_cyberbullying
                
                training_example = {
                    "text": text,
                    "category": "cyberbullying",
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"twitter_cyberbullying_{dataset_name.replace('/', '_')}",
                    "is_safe": is_safe
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            
            # Save to file
            output_file = f"data/processed/twitter_cyberbullying_{dataset_name.replace('/', '_')}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            return extracted_data
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No Twitter cyberbullying datasets could be loaded")
    return []

def main():
    """Load all Twitter datasets"""
    print("🚀 Loading Twitter Datasets for Content Moderation")
    print("=" * 50)
    
    # Load hate speech data
    hate_speech_data = load_twitter_hate_speech()
    
    print("\n" + "=" * 50)
    
    # Load cyberbullying data
    cyberbullying_data = load_twitter_cyberbullying()
    
    print("\n" + "=" * 50)
    
    # Summary
    total_samples = len(hate_speech_data) + len(cyberbullying_data)
    print(f"🎉 Twitter Data Loading Complete!")
    print(f"📊 Total samples extracted: {total_samples}")
    print(f"   - Hate Speech: {len(hate_speech_data)}")
    print(f"   - Cyberbullying: {len(cyberbullying_data)}")
    
    if total_samples > 0:
        print(f"💾 Data saved to data/processed/ directory")
        print(f"🚀 You can now combine this with your existing data!")

if __name__ == "__main__":
    main()
