#!/usr/bin/env python3
"""
Load toxic comment datasets for content moderation training
"""

import datasets
import json
import os
from typing import Dict, List
from tqdm import tqdm

def load_toxic_comments():
    """Try multiple toxic comment datasets"""
    print("🔍 Loading Toxic Comment Datasets...")
    
    # List of possible toxic comment datasets
    dataset_options = [
        "martin-ha/toxic-comment-classification",
        "toxic-comment-classification",
        "toxic_comments",
        "jigsaw_toxic",
        "toxic",
        "hate_speech_offensive",
        "offensive_language"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name)
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            for item in tqdm(dataset["train"], desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 8000:  # Limit for testing
                    break
                    
                text = item.get("text", "") or item.get("comment", "") or item.get("tweet", "")
                if not text:
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
                    category = "general"
                
                training_example = {
                    "text": text,
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"toxic_comments_{dataset_name.replace('/', '_')}",
                    "is_safe": is_safe,
                    "toxicity_scores": {
                        "toxic": toxic, "severe_toxic": severe_toxic,
                        "obscene": obscene, "threat": threat,
                        "insult": insult, "identity_hate": identity_hate
                    }
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            
            # Save to file
            output_file = f"data/processed/toxic_comments_{dataset_name.replace('/', '_')}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            return extracted_data
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No toxic comment datasets could be loaded")
    return []

def load_safebench():
    """Try multiple SafeBench datasets"""
    print("🔍 Loading SafeBench Datasets...")
    
    # List of possible SafeBench datasets
    dataset_options = [
        "safe-bench/safe-bench",
        "safe_bench",
        "safebench",
        "safety_benchmark",
        "safety_evaluation"
    ]
    
    for dataset_name in dataset_options:
        try:
            print(f"Trying: {dataset_name}")
            dataset = datasets.load_dataset(dataset_name)
            print(f"✅ Successfully loaded: {dataset_name}")
            
            # Extract data
            extracted_data = []
            for item in tqdm(dataset["train"], desc=f"Processing {dataset_name}"):
                if len(extracted_data) >= 8000:  # Limit for testing
                    break
                    
                text = item.get("prompt", "") or item.get("text", "") or item.get("question", "")
                if not text:
                    continue
                
                # Try different label fields
                label = item.get("label", item.get("safety_label", item.get("is_safe", "safe")))
                is_safe = label == "safe" or label == True or label == 1
                
                # Map categories
                category = item.get("category", "general")
                
                training_example = {
                    "text": text,
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": f"safebench_{dataset_name.replace('/', '_')}",
                    "is_safe": is_safe
                }
                extracted_data.append(training_example)
            
            print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
            
            # Save to file
            output_file = f"data/processed/safebench_{dataset_name.replace('/', '_')}.jsonl"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in extracted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved to: {output_file}")
            return extracted_data
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
    
    print("❌ No SafeBench datasets could be loaded")
    return []

def main():
    """Load all toxic and safety datasets"""
    print("🚀 Loading Toxic Comment and Safety Datasets")
    print("=" * 50)
    
    # Load toxic comments data
    toxic_data = load_toxic_comments()
    
    print("\n" + "=" * 50)
    
    # Load SafeBench data
    safebench_data = load_safebench()
    
    print("\n" + "=" * 50)
    
    # Summary
    total_samples = len(toxic_data) + len(safebench_data)
    print(f"🎉 Toxic and Safety Data Loading Complete!")
    print(f"📊 Total samples extracted: {total_samples}")
    print(f"   - Toxic Comments: {len(toxic_data)}")
    print(f"   - SafeBench: {len(safebench_data)}")
    
    if total_samples > 0:
        print(f"💾 Data saved to data/processed/ directory")
        print(f"🚀 You can now combine this with your existing data!")

if __name__ == "__main__":
    main()
