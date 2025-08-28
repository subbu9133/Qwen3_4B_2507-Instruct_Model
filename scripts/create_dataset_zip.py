#!/usr/bin/env python3
"""
Create Dataset ZIP Archive
Compresses the MASSIVE dataset into a .zip file for easy storage and sharing
"""

import os
import zipfile
import json
from pathlib import Path
from datetime import datetime

def create_dataset_zip():
    """Create a ZIP archive of the MASSIVE dataset"""
    print("Creating Dataset ZIP Archive")
    print("=" * 50)
    
    data_dir = Path("data/processed")
    
    # Create ZIP filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"Qwen3_4B_MASSIVE_Dataset_{timestamp}.zip"
    
    print(f"Creating ZIP file: {zip_filename}")
    
    # Files to include in the ZIP
    files_to_zip = [
        "MASSIVE_combined_train.jsonl",
        "MASSIVE_combined_validation.jsonl"
    ]
    
    # Also include individual collector files for reference
    collector_files = list(data_dir.glob("collector*_*.jsonl"))
    
    # Create the ZIP file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print(f"\n📦 Adding MASSIVE combined dataset...")
        
        # Add main dataset files
        for file_name in files_to_zip:
            file_path = data_dir / file_name
            if file_path.exists():
                zipf.write(file_path, f"dataset/{file_name}")
                print(f"   ✅ Added: {file_name}")
            else:
                print(f"   ❌ Not found: {file_name}")
        
        print(f"\n📁 Adding individual collector files...")
        
        # Add collector files
        for file_path in collector_files:
            zipf.write(file_path, f"collectors/{file_path.name}")
            print(f"   ✅ Added: {file_path.name}")
        
        # Add dataset statistics and info
        print(f"\n📊 Adding dataset information...")
        
        # Create dataset info file
        info_data = {
            "dataset_name": "Qwen3-4B Content Moderation MASSIVE Dataset",
            "created_date": timestamp,
            "total_samples": 75000,
            "training_samples": 41236,
            "validation_samples": 4582,
            "balanced_samples": 45818,
            "data_sources": [
                "Airoboros Safety Dataset",
                "OpenOrca Instruction Following",
                "Hate Speech Detection",
                "Toxicity Classification", 
                "Review Sentiment Analysis",
                "Twitter Sentiment Data",
                "Amazon Review Safety"
            ],
            "file_structure": {
                "dataset/": "Main training and validation files",
                "collectors/": "Individual collector output files"
            },
            "usage": "Fine-tune Qwen3-4B model for content moderation tasks"
        }
        
        info_file = "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        
        zipf.write(info_file, "dataset_info.json")
        print(f"   ✅ Added: dataset_info.json")
        
        # Clean up temporary info file
        os.remove(info_file)
        
        # Add README file
        readme_content = """# Qwen3-4B Content Moderation MASSIVE Dataset

## Overview
This dataset contains 75,000 samples for training a Qwen3-4B model for content moderation tasks.

## Dataset Statistics
- Total samples: 75,000
- Training samples: 41,236
- Validation samples: 4,582
- Balanced samples: 45,818 (1:1 safe/unsafe ratio)

## Data Sources
1. Airoboros Safety Dataset
2. OpenOrca Instruction Following
3. Hate Speech Detection
4. Toxicity Classification
5. Review Sentiment Analysis
6. Twitter Sentiment Data
7. Amazon Review Safety

## File Structure
- `dataset/` - Main training and validation files
- `collectors/` - Individual collector output files
- `dataset_info.json` - Dataset metadata

## Usage
Use this dataset to fine-tune your Qwen3-4B model for content moderation tasks.
The model will learn to classify text as SAFE or UNSAFE based on various safety criteria.

## Model Target
This dataset is designed to create a content moderation model similar to:
- Google's ShieldGemma-2B
- Meta's Llama-Guard-4-12B

## Created
Generated on: {timestamp}
""".format(timestamp=timestamp)
        
        readme_file = "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        zipf.write(readme_file, "README.md")
        print(f"   ✅ Added: README.md")
        
        # Clean up temporary README file
        os.remove(readme_file)
    
    # Get ZIP file size
    zip_size = os.path.getsize(zip_filename)
    zip_size_mb = zip_size / (1024 * 1024)
    
    print(f"\n🎉 Dataset ZIP Archive Created Successfully!")
    print(f"=" * 50)
    print(f"📁 ZIP File: {zip_filename}")
    print(f"📊 Size: {zip_size_mb:.2f} MB")
    print(f"📦 Contents:")
    print(f"   - MASSIVE combined dataset (training + validation)")
    print(f"   - Individual collector files")
    print(f"   - Dataset information and documentation")
    
    print(f"\n💾 Your MASSIVE dataset is now compressed and ready for:")
    print(f"   - Easy storage and backup")
    print(f"   - Sharing with team members")
    print(f"   - Uploading to cloud storage")
    print(f"   - Distribution to other researchers")
    
    return zip_filename, zip_size_mb

def main():
    print("🚀 Creating ZIP Archive of MASSIVE Dataset")
    print("=" * 60)
    
    try:
        zip_filename, zip_size = create_dataset_zip()
        
        print(f"\n✅ ZIP Archive Complete!")
        print(f"📁 File: {zip_filename}")
        print(f"📊 Size: {zip_size:.2f} MB")
        print(f"\n🎯 Your MASSIVE dataset is now portable and shareable!")
        
    except Exception as e:
        print(f"❌ Error creating ZIP archive: {e}")

if __name__ == "__main__":
    main()
