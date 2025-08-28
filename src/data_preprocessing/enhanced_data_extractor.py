import datasets
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import re

class EnhancedContentModerationDataExtractor:
    """Enhanced extractor with Twitter and Amazon datasets for stronger content moderation"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_airoboros_safety(self, max_samples: int = 20000) -> List[Dict]:
        """Extract data from Airoboros safety dataset"""
        print("Loading Airoboros Safety Dataset...")
        try:
            # Try different possible dataset names
            dataset_names = [
                "jondurbin/airoboros-3.1-safety",
                "jondurbin/airoboros-3.1",
                "jondurbin/airoboros-safety"
            ]
            
            dataset = None
            for name in dataset_names:
                try:
                    print(f"Trying dataset: {name}")
                    dataset = datasets.load_dataset(name)
                    print(f"Successfully loaded: {name}")
                    break
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue
            
            if dataset is None:
                print("❌ Could not load any Airoboros dataset, skipping...")
                return []
                
        except Exception as e:
            print(f"❌ Error loading Airoboros dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing Airoboros"):
                if len(extracted_data) >= max_samples:
                    break
                    
                # Extract conversation text
                if "conversations" in item and len(item["conversations"]) > 0:
                    text = item["conversations"][0]["value"]
                    
                    # Determine safety category and label
                    category = item.get("category", "general")
                    is_safe = item.get("is_safe", True)
                    
                    # Create training example
                    training_example = {
                        "text": text,
                        "category": category,
                        "label": "SAFE" if is_safe else "UNSAFE",
                        "source": "airoboros",
                        "is_safe": is_safe
                    }
                    extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing Airoboros data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Airoboros")
        return extracted_data
    
    def extract_safebench(self, max_samples: int = 15000) -> List[Dict]:
        """Extract data from SafeBench dataset"""
        print("Loading SafeBench Dataset...")
        try:
            dataset = datasets.load_dataset("safe-bench/safe-bench")
        except Exception as e:
            print(f"❌ Error loading SafeBench dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing SafeBench"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("prompt", "")
                if not text:
                    continue
                    
                # Map SafeBench categories to our safety categories
                category = item.get("category", "general")
                label = item.get("label", "safe")
                is_safe = label == "safe"
                
                training_example = {
                    "text": text,
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": "safebench",
                    "is_safe": is_safe
                }
                extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing SafeBench data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from SafeBench")
        return extracted_data
    
    def extract_toxicity_prompts(self, max_samples: int = 10000) -> List[Dict]:
        """Extract data from RealToxicityPrompts dataset"""
        print("Loading RealToxicityPrompts Dataset...")
        try:
            dataset = datasets.load_dataset("allenai/real-toxicity-prompts")
        except Exception as e:
            print(f"❌ Error loading RealToxicityPrompts dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing Toxicity Prompts"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("text", "")
                if not text:
                    continue
                    
                # Use toxicity scores to determine safety
                toxicity_score = item.get("toxicity", 0.0)
                is_safe = toxicity_score < 0.5  # Threshold for safety
                
                training_example = {
                    "text": text,
                    "category": "toxicity",
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": "toxicity_prompts",
                    "is_safe": is_safe,
                    "toxicity_score": toxicity_score
                }
                extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing Toxicity Prompts data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Toxicity Prompts")
        return extracted_data
    
    def extract_openorca_safety(self, max_samples: int = 5000) -> List[Dict]:
        """Extract safety-related data from OpenOrca"""
        print("Loading OpenOrca Dataset...")
        try:
            dataset = datasets.load_dataset("Open-Orca/OpenOrca", split="train")
        except Exception as e:
            print(f"❌ Error loading OpenOrca dataset: {e}")
            return []
        
        extracted_data = []
        safety_keywords = [
            "harmful", "dangerous", "violent", "hate", "discrimination",
            "inappropriate", "unsafe", "toxic", "offensive"
        ]
        
        try:
            for item in tqdm(dataset, desc="Processing OpenOrca"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("question", "") + " " + item.get("response", "")
                if not text:
                    continue
                    
                # Check if text contains safety-related content
                contains_safety_content = any(keyword in text.lower() for keyword in safety_keywords)
                
                if contains_safety_content:
                    # Simple heuristic: if it contains safety keywords, mark as potentially unsafe
                    is_safe = False
                    category = "general"
                    
                    training_example = {
                        "text": text[:500],  # Limit length
                        "category": category,
                        "label": "SAFE" if is_safe else "UNSAFE",
                        "source": "openorca",
                        "is_safe": is_safe
                    }
                    extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing OpenOrca data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from OpenOrca")
        return extracted_data
        
    def extract_twitter_hate_speech(self, max_samples: int = 15000) -> List[Dict]:
        """Extract data from Twitter hate speech dataset"""
        print("Loading Twitter Hate Speech Dataset...")
        try:
            dataset = datasets.load_dataset("t-davidson/hate_speech_and_offensive_language")
        except Exception as e:
            print(f"❌ Error loading Twitter hate speech dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing Twitter Hate Speech"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("tweet", "")
                if not text:
                    continue
                
                # Map labels to safety
                label = item.get("class", 0)
                # 0: hate speech, 1: offensive language, 2: neither
                is_safe = label == 2
                
                # Map to specific categories
                if label == 0:
                    category = "hate_speech"
                elif label == 1:
                    category = "offensive_language"
                else:
                    category = "general"
                
                training_example = {
                    "text": text,
                    "category": category,
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": "twitter_hate_speech",
                    "is_safe": is_safe,
                    "original_label": label
                }
                extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing Twitter hate speech data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Twitter Hate Speech")
        return extracted_data
    
    def extract_twitter_cyberbullying(self, max_samples: int = 10000) -> List[Dict]:
        """Extract data from Twitter cyberbullying dataset"""
        print("Loading Twitter Cyberbullying Dataset...")
        try:
            dataset = datasets.load_dataset("cyberbullying_detection/cyberbullying_tweets")
        except Exception as e:
            print(f"❌ Error loading Twitter cyberbullying dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing Twitter Cyberbullying"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("tweet_text", "")
                if not text:
                    continue
                
                # Check if it's cyberbullying
                is_cyberbullying = item.get("cyberbullying_type", "not_cyberbullying") != "not_cyberbullying"
                is_safe = not is_cyberbullying
                
                training_example = {
                    "text": text,
                    "category": "cyberbullying",
                    "label": "SAFE" if is_safe else "UNSAFE",
                    "source": "twitter_cyberbullying",
                    "is_safe": is_safe,
                    "cyberbullying_type": item.get("cyberbullying_type", "not_cyberbullying")
                }
                extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing Twitter cyberbullying data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Twitter Cyberbullying")
        return extracted_data
    
    def extract_amazon_reviews_safety(self, max_samples: int = 8000) -> List[Dict]:
        """Extract safety-related data from Amazon reviews"""
        print("Loading Amazon Reviews Safety Dataset...")
        try:
            # Try to load Amazon reviews dataset
            dataset = datasets.load_dataset("amazon_us_reviews", "Books_v1_02", split="train")
        except Exception as e:
            print(f"❌ Error loading Amazon reviews dataset: {e}")
            return []
        
        extracted_data = []
        safety_keywords = [
            "harmful", "dangerous", "inappropriate", "offensive", "toxic",
            "hate", "violence", "discrimination", "harassment", "bullying"
        ]
        
        try:
            for item in tqdm(dataset, desc="Processing Amazon Reviews"):
                if len(extracted_data) >= max_samples:
                    break
                    
                review_text = item.get("review_body", "")
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
                        "source": "amazon_reviews",
                        "is_safe": is_safe,
                        "rating": item.get("star_rating", 0)
                    }
                    extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing Amazon reviews data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Amazon Reviews")
        return extracted_data
    
    def extract_toxic_comments(self, max_samples: int = 12000) -> List[Dict]:
        """Extract data from toxic comment classification dataset"""
        print("Loading Toxic Comments Dataset...")
        try:
            dataset = datasets.load_dataset("martin-ha/toxic-comment-classification")
        except Exception as e:
            print(f"❌ Error loading toxic comments dataset: {e}")
            return []
        
        extracted_data = []
        try:
            for item in tqdm(dataset["train"], desc="Processing Toxic Comments"):
                if len(extracted_data) >= max_samples:
                    break
                    
                text = item.get("text", "")
                if not text:
                    continue
                
                # Check toxicity labels
                toxic = item.get("toxic", 0)
                severe_toxic = item.get("severe_toxic", 0)
                obscene = item.get("obscene", 0)
                threat = item.get("threat", 0)
                insult = item.get("insult", 0)
                identity_hate = item.get("identity_hate", 0)
                
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
                    "source": "toxic_comments",
                    "is_safe": is_safe,
                    "toxicity_scores": {
                        "toxic": toxic, "severe_toxic": severe_toxic,
                        "obscene": obscene, "threat": threat,
                        "insult": insult, "identity_hate": identity_hate
                    }
                }
                extracted_data.append(training_example)
        except Exception as e:
            print(f"❌ Error processing toxic comments data: {e}")
        
        print(f"Extracted {len(extracted_data)} samples from Toxic Comments")
        return extracted_data
    
    def create_synthetic_safety_data(self, max_samples: int = 10000) -> List[Dict]:
        """Create synthetic safety training data when datasets are unavailable"""
        print("Creating synthetic safety training data...")
        
        # Safe examples
        safe_examples = [
            "Hello, how are you today?",
            "The weather is beautiful today.",
            "I love reading books and learning new things.",
            "Cooking is a wonderful hobby that brings people together.",
            "Music has the power to heal and inspire.",
            "Nature provides us with beauty and tranquility.",
            "Education opens doors to new opportunities.",
            "Friendship is one of life's greatest gifts.",
            "Helping others brings joy and fulfillment.",
            "Creativity allows us to express ourselves uniquely."
        ]
        
        # Unsafe examples (harmful content)
        unsafe_examples = [
            "I want to hurt someone badly.",
            "Let's spread hate and discrimination.",
            "How to make dangerous weapons at home.",
            "Ways to harm yourself and others.",
            "Promoting violence and aggression.",
            "Spreading false information and lies.",
            "Harassing and bullying people online.",
            "Sharing private information without consent.",
            "Encouraging illegal activities.",
            "Promoting harmful stereotypes and bias."
        ]
        
        extracted_data = []
        
        # Add safe examples
        for i, text in enumerate(safe_examples):
            if len(extracted_data) >= max_samples // 2:
                break
            extracted_data.append({
                "text": text,
                "category": "general",
                "label": "SAFE",
                "source": "synthetic",
                "is_safe": True
            })
        
        # Add unsafe examples
        for i, text in enumerate(unsafe_examples):
            if len(extracted_data) >= max_samples:
                break
            extracted_data.append({
                "text": text,
                "category": "general",
                "label": "UNSAFE",
                "source": "synthetic",
                "is_safe": False
            })
        
        print(f"Created {len(extracted_data)} synthetic safety examples")
        return extracted_data
    
    def create_training_prompts(self, data: List[Dict]) -> List[Dict]:
        """Convert extracted data into training prompts"""
        print("Creating training prompts...")
        
        training_data = []
        for item in tqdm(data, desc="Creating prompts"):
            # Create the safety classification prompt
            prompt = f"""<|im_start|>system
You are a content moderation expert. Analyze the following text and classify it as SAFE or UNSAFE based on the safety category.

Safety Category: {item['category']}
Text: {item['text']}

Please respond with only SAFE or UNSAFE.<|im_end|>
<|im_start|>user
Is this text safe for the category {item['category']}?<|im_end|>
<|im_start|>assistant
{item['label']}<|im_end|>"""
            
            training_data.append({
                "text": prompt,
                "label": item["label"],
                "category": item["category"],
                "source": item["source"],
                "is_safe": item["is_safe"]
            })
        
        return training_data
    
    def balance_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Balance the dataset between safe and unsafe examples"""
        safe_examples = [item for item in data if item["is_safe"]]
        unsafe_examples = [item for item in data if not item["is_safe"]]
        
        print(f"Safe examples: {len(safe_examples)}")
        print(f"Unsafe examples: {len(unsafe_examples)}")
        
        # Balance by undersampling the majority class
        min_count = min(len(safe_examples), len(unsafe_examples))
        
        if len(safe_examples) > min_count:
            import random
            safe_examples = random.sample(safe_examples, min_count)
        
        if len(unsafe_examples) > min_count:
            import random
            unsafe_examples = random.sample(unsafe_examples, min_count)
        
        balanced_data = safe_examples + unsafe_examples
        
        # Shuffle the data
        import random
        random.shuffle(balanced_data)
        
        # Split into train/validation (90/10)
        split_idx = int(len(balanced_data) * 0.9)
        train_data = balanced_data[:split_idx]
        val_data = balanced_data[split_idx:]
        
        print(f"Balanced dataset - Train: {len(train_data)}, Validation: {len(val_data)}")
        return train_data, val_data
    
    def save_data(self, train_data: List[Dict], val_data: List[Dict]):
        """Save processed data to JSONL files"""
        print("Saving processed data...")
        
        # Save training data
        train_file = os.path.join(self.output_dir, "train.jsonl")
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Save validation data
        val_file = os.path.join(self.output_dir, "validation.jsonl")
        with open(val_file, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Data saved to {self.output_dir}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
    
    def run_enhanced_extraction_pipeline(self):
        """Run the enhanced data extraction pipeline with Twitter and Amazon data"""
        print("🚀 Starting Enhanced Content Moderation Data Extraction Pipeline")
        print("📊 Including Twitter and Amazon datasets for stronger training...")
        
        # Extract from all sources including new ones
        airoboros_data = self.extract_airoboros_safety()
        safebench_data = self.extract_safebench()
        toxicity_data = self.extract_toxicity_prompts()
        openorca_data = self.extract_openorca_safety()
        
        # NEW: Extract from Twitter and Amazon datasets
        twitter_hate_data = self.extract_twitter_hate_speech()
        twitter_bullying_data = self.extract_twitter_cyberbullying()
        amazon_reviews_data = self.extract_amazon_reviews_safety()
        toxic_comments_data = self.extract_toxic_comments()
        
        # Combine all data
        all_data = (
            airoboros_data + safebench_data + toxicity_data + openorca_data +
            twitter_hate_data + twitter_bullying_data + amazon_reviews_data + toxic_comments_data
        )
        
        print(f"\n📈 Total Enhanced Dataset: {len(all_data)} samples")
        print(f"📊 Breakdown by source:")
        print(f"   - Airoboros: {len(airoboros_data)}")
        print(f"   - SafeBench: {len(safebench_data)}")
        print(f"   - Toxicity: {len(toxicity_data)}")
        print(f"   - OpenOrca: {len(openorca_data)}")
        print(f"   - Twitter Hate Speech: {len(twitter_hate_data)}")
        print(f"   - Twitter Cyberbullying: {len(twitter_bullying_data)}")
        print(f"   - Amazon Reviews: {len(amazon_reviews_data)}")
        print(f"   - Toxic Comments: {len(toxic_comments_data)}")
        
        if len(all_data) == 0:
            print("❌ No data could be extracted. Creating synthetic safety data...")
            all_data = self.create_synthetic_safety_data(max_samples=2000)
        
        # Create training prompts
        training_data = self.create_training_prompts(all_data)
        
        # Balance and split dataset
        train_data, val_data = self.balance_dataset(training_data)
        
        # Save processed data
        self.save_data(train_data, val_data)
        
        print("\n🎉 Enhanced data extraction pipeline completed successfully!")
        print(f"🚀 Your model will now be MUCH stronger with real-world data!")
        return train_data, val_data

if __name__ == "__main__":
    extractor = EnhancedContentModerationDataExtractor()
    extractor.run_enhanced_extraction_pipeline()
