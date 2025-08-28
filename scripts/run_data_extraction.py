#!/usr/bin/env python3
"""
Run only the data extraction pipeline for Qwen3-4B content moderation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Run the data extraction pipeline"""
    print("🚀 Starting Qwen3-4B Content Moderation Data Extraction")
    
    try:
        from data_preprocessing.data_extractor import ContentModerationDataExtractor
        
        # Initialize extractor
        extractor = ContentModerationDataExtractor()
        
        # Run extraction pipeline
        train_data, val_data = extractor.run_extraction_pipeline()
        
        print("\n✅ Data extraction completed successfully!")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        print(f"💾 Data saved to: data/processed/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
