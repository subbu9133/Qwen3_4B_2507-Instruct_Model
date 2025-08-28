#!/usr/bin/env python3
"""
Collector 5: Twitter Datasets
Uses twitter_processor.py to collect Twitter sentiment data
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to path to import the processor
sys.path.append(str(Path(__file__).parent.parent))

def collect_twitter_datasets():
    """Collect Twitter data using the dedicated processor"""
    print("🔍 Collector 5: Twitter Datasets")
    print("=" * 50)
    print("Using twitter_processor.py for data collection")
    print("=" * 50)
    
    try:
        # Import the Twitter processor
        from twitter_processor import TwitterSentimentProcessor
        
        # Initialize processor
        processor = TwitterSentimentProcessor()
        
        # Try to load Twitter datasets from Hugging Face first
        print("\n🔄 Trying to load Twitter datasets from Hugging Face...")
        
        try:
            # Try to load a working Twitter dataset from Hugging Face
            from datasets import load_dataset
            
            # Try different Twitter datasets that are known to work
            twitter_datasets = [
                ("tweet_eval", "sentiment"),  # Working config
                ("tweet_eval", "hate"),       # Working config  
                ("tweet_eval", "offensive"),  # Working config
                ("cardiffnlp/tweet_sentiment_multilingual", None),  # Try without config
                ("sentiment140", None)        # Try without config
            ]
            
            for dataset_name, config in twitter_datasets:
                try:
                    if config:
                        print(f"\n🔍 Trying: {dataset_name} with config: {config}")
                        dataset = load_dataset(dataset_name, config, split="train")
                    else:
                        print(f"\n🔍 Trying: {dataset_name}")
                        dataset = load_dataset(dataset_name, split="train")
                    
                    print(f"✅ Successfully loaded: {dataset_name}")
                    
                    # Extract data directly (simplified approach)
                    extracted_data = []
                    max_samples = 8000
                    
                    for i, item in enumerate(dataset):
                        if len(extracted_data) >= max_samples:
                            break
                        
                        # Get text from various possible fields
                        text = item.get("text", "") or item.get("tweet", "") or item.get("sentence", "")
                        if not text or len(text) < 5:
                            continue
                        
                        # Get label/sentiment
                        label = item.get("label", item.get("sentiment", item.get("class", 0)))
                        
                        # Map to safety classification based on dataset type
                        if config == "hate" or config == "offensive":
                            # For hate/offensive datasets, label 0 is usually safe
                            is_safe = label == 0
                        elif config == "sentiment":
                            # For sentiment datasets, label 0 is usually negative
                            is_safe = label == 1  # 1 = positive
                        else:
                            # Default mapping
                            if isinstance(label, str):
                                label = label.lower()
                                is_safe = label in ["positive", "good", "safe", "0"]
                            else:
                                # Numeric labels: assume 0=negative, 1=positive
                                is_safe = label == 1
                        
                        # Map to categories
                        if is_safe:
                            category = "twitter_positive_sentiment"
                        else:
                            category = "twitter_negative_sentiment"
                        
                        training_example = {
                            "text": text[:1000],  # Limit length
                            "category": category,
                            "label": "SAFE" if is_safe else "UNSAFE",
                            "source": f"twitter_{dataset_name}_{config}" if config else f"twitter_{dataset_name}",
                            "is_safe": is_safe,
                            "original_label": str(label),
                            "platform": "twitter"
                        }
                        extracted_data.append(training_example)
                    
                    print(f"📊 Extracted {len(extracted_data)} samples from {dataset_name}")
                    
                    if len(extracted_data) > 0:
                        # Save to file
                        config_suffix = f"_{config}" if config else ""
                        output_file = f"data/processed/collector5_twitter_{dataset_name.replace('/', '_')}{config_suffix}.jsonl"
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        with open(output_file, "w", encoding="utf-8") as f:
                            for item in extracted_data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
                        print(f"💾 Saved to: {output_file}")
                        
                        print(f"\n🎉 Twitter Collection Complete!")
                        print(f"📊 Total samples extracted: {len(extracted_data)}")
                        return len(extracted_data)
                    
                except Exception as e:
                    print(f"❌ Failed to load {dataset_name}: {e}")
                    continue
            
            # If no Hugging Face datasets worked, show instructions
            print("\n❌ No Hugging Face Twitter datasets could be loaded")
            print("💡 Falling back to processor instructions...")
            
        except ImportError:
            print("❌ Hugging Face datasets library not available")
            print("💡 Install with: pip install datasets")
        
        # Show download instructions as fallback
        processor.download_dataset_info()
        
        print("\n📥 To collect Twitter data, you need to:")
        print("1. Download a Twitter dataset (e.g., Sentiment140 from Kaggle)")
        print("2. Place it in data/raw/ directory")
        print("3. Run the processor with the file path")
        
        # Try to find existing Twitter datasets in data/raw/
        raw_dir = Path("data/raw")
        twitter_files = []
        
        if raw_dir.exists():
            for file in raw_dir.glob("*"):
                if any(keyword in file.name.lower() for keyword in ['twitter', 'sentiment', 'tweet']):
                    twitter_files.append(file)
        
        if twitter_files:
            print(f"\n📁 Found {len(twitter_files)} potential Twitter datasets:")
            for file in twitter_files:
                print(f"   - {file.name}")
            
            # Process the first found file
            selected_file = twitter_files[0]
            print(f"\n🔄 Processing: {selected_file.name}")
            
            # Determine dataset type and run processor
            if 'sentiment140' in selected_file.name.lower():
                dataset_type = "sentiment140"
            else:
                dataset_type = "generic"
            
            output_files = processor.process_full_pipeline(
                dataset_type=dataset_type,
                file_path=str(selected_file),
                sample_size=10000,  # Limit for content moderation
                output_dir="data/processed"
            )
            
            # Convert sentiment format to content moderation format
            print("\n🔄 Converting to content moderation format...")
            convert_twitter_to_moderation(output_files)
            
            print(f"\n🎉 Twitter Collection Complete!")
            print(f"📊 Data processed and converted to content moderation format")
            return 10000  # Return estimated count
            
        else:
            print("\n❌ No Twitter datasets found in data/raw/")
            print("💡 Please download a Twitter dataset and place it in data/raw/")
            print("   Recommended: Sentiment140 from Kaggle")
            return 0
            
    except ImportError as e:
        print(f"❌ Error importing twitter_processor: {e}")
        print("💡 Make sure twitter_processor.py is in the project root")
        return 0
    except Exception as e:
        print(f"❌ Error in Twitter collection: {e}")
        return 0

def convert_twitter_to_moderation(output_files):
    """Convert Twitter sentiment data to content moderation format"""
    try:
        # Read the processed Twitter data
        train_file = output_files.get('train')
        if not train_file or not Path(train_file).exists():
            print("❌ Training file not found")
            return
        
        # Read and convert data
        converted_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    
                    # Convert sentiment to safety classification
                    # Negative sentiment = potentially unsafe content
                    is_safe = item.get('sentiment') != 'negative'
                    
                    # Map to content moderation categories
                    if item.get('sentiment') == 'negative':
                        category = 'twitter_negative_sentiment'
                    elif item.get('sentiment') == 'positive':
                        category = 'twitter_positive_sentiment'
                    else:
                        category = 'twitter_neutral_sentiment'
                    
                    # Create content moderation example
                    moderation_example = {
                        "text": item.get('text', '')[:1000],  # Limit length
                        "category": category,
                        "label": "SAFE" if is_safe else "UNSAFE",
                        "source": "twitter_processor",
                        "is_safe": is_safe,
                        "original_sentiment": item.get('sentiment'),
                        "platform": "twitter",
                        "confidence": item.get('confidence', 0.7)
                    }
                    converted_data.append(moderation_example)
        
        # Save converted data
        output_file = "data/processed/collector5_twitter_processed.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"💾 Converted {len(converted_data)} samples to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error converting Twitter data: {e}")

if __name__ == "__main__":
    collect_twitter_datasets()
