import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# ============================================================================
# LOAD YOUR TRAINED MODEL
# ============================================================================

print("🤖 Loading trained model...")
model_path = './results/checkpoint-200'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

label_names = [
    'Computer Science', 
    'Physics', 
    'Mathematics', 
    'Statistics', 
    'Quantitative Biology', 
    'Quantitative Finance'
]

print("✅ Model loaded successfully!\n")

# ============================================================================
# FUNCTION TO CLASSIFY A NEW PAPER
# ============================================================================

def classify_paper(abstract_text, threshold=0.5):
    """
    Classify a scientific paper abstract into multiple categories
    
    Args:
        abstract_text (str): The paper's abstract text
        threshold (float): Confidence threshold (0.5 = 50%)
    
    Returns:
        dict: Predictions with labels and confidence scores
    """
    # Tokenize the input
    inputs = tokenizer(
        abstract_text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding=True
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply sigmoid to get probabilities (for multi-label)
        probabilities = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    # Get predictions based on threshold
    predictions = (probabilities >= threshold).astype(int)
    
    # Format results
    results = {
        'predicted_labels': [label_names[i] for i in range(len(predictions)) if predictions[i] == 1],
        'all_probabilities': {label_names[i]: float(probabilities[i]) for i in range(len(label_names))},
        'confidence_scores': {label_names[i]: float(probabilities[i]) for i in range(len(predictions)) if predictions[i] == 1}
    }
    
    return results

# ============================================================================
# EXAMPLE 1: CLASSIFY A SINGLE PAPER
# ============================================================================

print("="*70)
print("EXAMPLE 1: Classify a Single Paper")
print("="*70)

# Example paper abstract (you can replace this with any new paper)
new_paper = """
Deep learning has revolutionized computer vision through convolutional neural 
networks. We present a novel architecture that combines attention mechanisms 
with residual connections to improve image classification accuracy. Our method 
achieves state-of-the-art results on ImageNet with fewer parameters than 
existing approaches. We also provide mathematical proofs of convergence.
"""

print("\n📄 Paper Abstract:")
print(new_paper.strip())

result = classify_paper(new_paper)

print(f"\n🎯 PREDICTED CATEGORIES:")
if result['predicted_labels']:
    for label in result['predicted_labels']:
        confidence = result['confidence_scores'][label]
        print(f"   ✓ {label}: {confidence:.2%} confidence")
else:
    print("   No categories predicted above threshold")

print(f"\n📊 ALL CATEGORY PROBABILITIES:")
for label, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 20)
    print(f"   {label:25s} {prob:.2%} {bar}")

# ============================================================================
# EXAMPLE 2: CLASSIFY MULTIPLE PAPERS FROM CSV
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Classify Multiple Papers from CSV")
print("="*70)

# Load your CSV
df = pd.read_csv('train.csv')

# Take 3 random papers to classify
test_papers = df.sample(n=3, random_state=123)

for idx, row in test_papers.iterrows():
    print(f"\n{'='*70}")
    print(f"Paper #{idx}")
    print(f"{'='*70}")
    print(f"Title: {row['TITLE']}")
    print(f"Abstract: {row['ABSTRACT'][:200]}...")
    
    # Classify
    result = classify_paper(row['ABSTRACT'])
    
    print(f"\n🎯 Predicted: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None'}")
    
    # Show true labels
    true_labels = [label_names[i] for i in range(len(label_names)) if row[label_names[i]] == 1]
    print(f"✓ Actual:    {', '.join(true_labels)}")
    
    # Check if correct
    if set(result['predicted_labels']) == set(true_labels):
        print("🎉 PERFECT MATCH!")
    else:
        print("⚠️  Partial match or mismatch")

# ============================================================================
# EXAMPLE 3: CLASSIFY FROM USER INPUT (Interactive)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Interactive Classification")
print("="*70)

def interactive_classify():
    """
    Interactive mode - paste any paper abstract
    """
    print("\n💡 Paste a paper abstract below (or type 'quit' to exit):")
    print("   (Press Enter twice when done)\n")
    
    while True:
        lines = []
        print("Abstract: ", end="")
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        
        abstract = " ".join(lines).strip()
        
        if abstract.lower() == 'quit' or abstract == "":
            print("👋 Exiting...")
            break
        
        result = classify_paper(abstract)
        
        print(f"\n🎯 RESULTS:")
        print(f"   Categories: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None detected'}")
        print(f"\n   Top 3 most likely:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for label, prob in sorted_probs:
            print(f"      • {label}: {prob:.2%}")
        
        print("\n" + "-"*70)

# Uncomment below to enable interactive mode
# interactive_classify()

# ============================================================================
# EXAMPLE 4: CLASSIFY FROM A TEXT FILE
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Classify from Text File")
print("="*70)

def classify_from_file(file_path):
    """
    Read a paper abstract from a text file and classify it
    
    Args:
        file_path (str): Path to text file containing the abstract
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            abstract = f.read().strip()
        
        print(f"\n📄 Reading from: {file_path}")
        print(f"Abstract preview: {abstract[:200]}...\n")
        
        result = classify_paper(abstract)
        
        print("🎯 CLASSIFICATION RESULTS:")
        print(f"   Predicted Categories: {', '.join(result['predicted_labels'])}")
        print(f"\n   Confidence Scores:")
        for label in result['predicted_labels']:
            print(f"      • {label}: {result['confidence_scores'][label]:.2%}")
        
        return result
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

# Example usage (create a file called 'test_paper.txt' with an abstract):
# result = classify_from_file('test_paper.txt')

# ============================================================================
# EXAMPLE 5: BATCH CLASSIFY AND SAVE TO CSV
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Batch Classification - Save Results to CSV")
print("="*70)

def batch_classify_and_save(input_csv, output_csv, text_column='ABSTRACT'):
    """
    Classify all papers in a CSV and save results
    
    Args:
        input_csv (str): Input CSV file path
        output_csv (str): Output CSV file path
        text_column (str): Column name containing the abstract text
    """
    df = pd.read_csv(input_csv)
    
    print(f"\n📊 Classifying {len(df)} papers...")
    
    predictions_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"   Progress: {idx}/{len(df)}")
        
        result = classify_paper(row[text_column])
        
        # Add predictions to dataframe
        pred_row = {
            'ID': row['ID'],
            'TITLE': row['TITLE'],
            'ABSTRACT': row[text_column],
        }
        
        # Add binary predictions for each category
        for label in label_names:
            pred_row[f'Predicted_{label}'] = 1 if label in result['predicted_labels'] else 0
            pred_row[f'Confidence_{label}'] = result['all_probabilities'][label]
        
        predictions_list.append(pred_row)
    
    # Save results
    results_df = pd.DataFrame(predictions_list)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Results saved to: {output_csv}")
    print(f"   Total papers classified: {len(results_df)}")
    
    return results_df

# Example: Classify first 50 papers and save
sample_df = df.head(50)
sample_df.to_csv('sample_papers.csv', index=False)
results = batch_classify_and_save('sample_papers.csv', 'classification_results.csv')

print(f"\n✅ Sample results saved!")
print(f"   Input: sample_papers.csv")
print(f"   Output: classification_results.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📋 USAGE SUMMARY")
print("="*70)

print("""
You now have 5 ways to use your trained model:

1️⃣  Single Paper Classification:
    result = classify_paper("Your abstract text here")
    
2️⃣  Classify from DataFrame:
    for idx, row in df.iterrows():
        result = classify_paper(row['ABSTRACT'])
    
3️⃣  Interactive Mode:
    interactive_classify()  # Uncomment to enable
    
4️⃣  From Text File:
    result = classify_from_file('paper.txt')
    
5️⃣  Batch Process CSV:
    batch_classify_and_save('input.csv', 'output.csv')

🎯 For your presentation demo:
   - Use Example 1 or 3 (interactive) 
   - Show live classification of a new paper!
""")

print("="*70)
print("✅ Ready to classify papers!")
print("="*70)