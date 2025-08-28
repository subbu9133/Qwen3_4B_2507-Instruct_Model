#!/usr/bin/env python3
"""
Run enhanced data extraction pipeline with Twitter and Amazon datasets
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Run the enhanced data extraction pipeline"""
    print("🚀 Starting Enhanced Qwen3-4B Content Moderation Data Extraction")
    print("📊 Including Twitter and Amazon datasets for MUCH stronger training!")
    
    try:
        from data_preprocessing.enhanced_data_extractor import EnhancedContentModerationDataExtractor
        
        # Initialize enhanced extractor
        extractor = EnhancedContentModerationDataExtractor()
        
        # Run enhanced extraction pipeline
        train_data, val_data = extractor.run_enhanced_extraction_pipeline()
        
        print("\n✅ Enhanced data extraction completed successfully!")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        print(f"💾 Data saved to: data/processed/")
        print(f"🚀 Your model will now be MUCH stronger with real-world data!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error during enhanced data extraction: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
