# import io
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer
# )
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
# import torch

# # ============================================================================
# # 1. LOAD DATASET FROM RAW CSV TEXT
# # ============================================================================

# csv_text = """ID,TITLE,ABSTRACT,Computer Science,Physics,Mathematics,Statistics,Quantitative Biology,Quantitative Finance
# 1,Reconstructing Subject-Specific Effect Maps," Predictive models allow subject-specific inference when analyzing disease related alterations in neuroimaging data. Given......",1,0,0,0,0,0
# 2,Quantum Entanglement in Systems,"This paper explores quantum entanglement phenomena in multi-particle systems using novel mathematical frameworks......",0,1,1,0,0,0
# 3,Statistical Methods for Big Data,"We present new statistical approaches for handling large-scale datasets with applications in machine learning......",1,0,0,1,0,0
# 4,Biological Network Analysis,"Network analysis techniques are applied to understand complex biological systems and protein interactions......",1,0,1,1,1,0
# 5,Financial Risk Modeling,"Advanced quantitative methods for modeling financial risk in volatile markets using stochastic processes......",0,0,1,1,0,1
# 6,Deep Learning Architectures,"Novel neural network architectures for computer vision tasks with improved performance and efficiency......",1,0,0,0,0,0
# 7,Particle Physics Simulations,"Simulation methods for particle collisions in high-energy physics experiments at CERN......",1,1,1,0,0,0
# 8,Bayesian Inference Methods,"Comprehensive study of Bayesian statistical methods for parameter estimation and hypothesis testing......",0,0,1,1,0,0"""

# # Load data using pandas
# df = pd.read_csv(io.StringIO(csv_text))

# print("Dataset loaded successfully!")
# print(f"Dataset shape: {df.shape}")
# print(f"\nFirst few rows:\n{df.head()}\n")

# # ============================================================================
# # 2. PREPARE DATA
# # ============================================================================

# # Define label columns
# label_columns = [
#     'Computer Science', 
#     'Physics', 
#     'Mathematics', 
#     'Statistics', 
#     'Quantitative Biology', 
#     'Quantitative Finance'
# ]

# # Extract text and labels
# texts = df['ABSTRACT'].tolist()
# labels = df[label_columns].values.astype(np.float32)

# print(f"Number of samples: {len(texts)}")
# print(f"Number of labels: {labels.shape[1]}")
# print(f"Label distribution:\n{df[label_columns].sum()}\n")

# # ============================================================================
# # 3. TRAIN/EVAL SPLIT
# # ============================================================================

# train_texts, eval_texts, train_labels, eval_labels = train_test_split(
#     texts, labels, test_size=0.2, random_state=42
# )

# print(f"Training samples: {len(train_texts)}")
# print(f"Evaluation samples: {len(eval_texts)}\n")

# # ============================================================================
# # 4. TOKENIZATION
# # ============================================================================

# model_name = 'bert-base-uncased'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def tokenize_function(examples):
#     """Tokenize the input texts"""
#     return tokenizer(
#         examples['text'],
#         max_length=128,
#         padding=True,
#         truncation=True
#     )

# # Create datasets
# train_dataset = Dataset.from_dict({
#     'text': train_texts,
#     'labels': train_labels.tolist()
# })

# eval_dataset = Dataset.from_dict({
#     'text': eval_texts,
#     'labels': eval_labels.tolist()
# })

# # Apply tokenization
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# # Set format for PyTorch
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# print("Tokenization completed!\n")

# # ============================================================================
# # 5. MODEL CONFIGURATION
# # ============================================================================

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=6,
#     problem_type="multi_label_classification"  # CRITICAL for multi-label
# )

# print(f"Model loaded: {model_name}")
# print(f"Problem type: multi_label_classification")
# print(f"Number of labels: 6\n")

# # ============================================================================
# # 6. COMPUTE METRICS FUNCTION
# # ============================================================================

# def compute_metrics(eval_pred):
#     """
#     Compute Macro-Averaged F1-Score and ROC-AUC for multi-label classification
#     """
#     logits, labels = eval_pred
    
#     # Apply sigmoid to get probabilities (for multi-label)
#     predictions = torch.sigmoid(torch.tensor(logits)).numpy()
    
#     # Convert probabilities to binary predictions (threshold = 0.5)
#     binary_predictions = (predictions > 0.5).astype(int)
    
#     # Calculate Macro F1-Score
#     macro_f1 = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    
#     # Calculate Macro ROC-AUC
#     try:
#         macro_roc_auc = roc_auc_score(labels, predictions, average='macro')
#     except ValueError:
#         # Handle case where some labels might not appear in eval set
#         macro_roc_auc = 0.0
    
#     return {
#         'macro_f1': macro_f1,
#         'macro_roc_auc': macro_roc_auc
#     }

# # ============================================================================
# # 7. TRAINING ARGUMENTS
# # ============================================================================

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="macro_f1",
#     logging_dir='./logs',
#     logging_steps=10,
#     seed=42,
#     report_to="none"  # Disable wandb/tensorboard for simplicity
# )

# # ============================================================================
# # 8. INITIALIZE TRAINER
# # ============================================================================

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer
# )

# print("=" * 70)
# print("STARTING TRAINING")
# print("=" * 70)

# # ============================================================================
# # 9. TRAIN THE MODEL
# # ============================================================================

# trainer.train()

# print("\n" + "=" * 70)
# print("TRAINING COMPLETED")
# print("=" * 70)

# # ============================================================================
# # 10. EVALUATE AND DISPLAY RESULTS
# # ============================================================================

# print("\n" + "=" * 70)
# print("FINAL EVALUATION RESULTS")
# print("=" * 70)

# eval_results = trainer.evaluate()

# print(f"\n{'Metric':<30} {'Value':<10}")
# print("-" * 45)
# print(f"{'Macro-Averaged F1-Score':<30} {eval_results['eval_macro_f1']:.4f}")
# print(f"{'Macro-Averaged ROC-AUC':<30} {eval_results['eval_macro_roc_auc']:.4f}")
# print(f"{'Evaluation Loss':<30} {eval_results['eval_loss']:.4f}")
# print("-" * 45)

# # ============================================================================
# # 11. ADDITIONAL ANALYSIS
# # ============================================================================

# print("\n" + "=" * 70)
# print("SAMPLE PREDICTIONS")
# print("=" * 70)

# # Get predictions on eval set
# predictions = trainer.predict(eval_dataset)
# pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
# pred_labels = (pred_probs > 0.5).astype(int)

# # Show first 3 samples
# for idx in range(min(3, len(eval_texts))):
#     print(f"\nSample {idx + 1}:")
#     print(f"Text: {eval_texts[idx][:100]}...")
#     print(f"True Labels: {eval_labels[idx].astype(int)}")
#     print(f"Predicted:   {pred_labels[idx]}")
#     print(f"Probabilities: {pred_probs[idx].round(3)}")

# print("\n" + "=" * 70)
# print("SCRIPT COMPLETED SUCCESSFULLY")
# print("=" * 70)

# ---------------------------------------
import io
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR NEEDS
# ============================================================================

USE_SAMPLE_SIZE = 10000  # Set to None to use full dataset (309k)
USE_FAST_MODEL = True    # True = DistilBERT (fast), False = BERT (slow)
NUM_EPOCHS = 1           # 1 epoch is often enough for large datasets
MAX_STEPS = 3000         # Stop after this many steps (remove limit: set to -1)
BATCH_SIZE = 16          # Increase to 32 or 64 if you have good GPU

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

# Replace this with your actual CSV loading:
# df = pd.read_csv('your_file.csv')

# For demo purposes (replace with your data):
csv_text = """ID,TITLE,ABSTRACT,Computer Science,Physics,Mathematics,Statistics,Quantitative Biology,Quantitative Finance
1,Reconstructing Subject-Specific Effect Maps," Predictive models allow subject-specific inference when analyzing disease related alterations in neuroimaging data. Given......",1,0,0,0,0,0"""

df = pd.read_csv(io.StringIO(csv_text))

print(f"Original dataset size: {len(df)} samples")

# ============================================================================
# OPTIMIZATION 1: Sample the dataset for faster training
# ============================================================================

if USE_SAMPLE_SIZE and USE_SAMPLE_SIZE < len(df):
    df = df.sample(n=USE_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Using sample of {USE_SAMPLE_SIZE} samples for faster training")
else:
    print(f"Using full dataset: {len(df)} samples")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================

label_columns = [
    'Computer Science', 'Physics', 'Mathematics', 
    'Statistics', 'Quantitative Biology', 'Quantitative Finance'
]

texts = df['ABSTRACT'].tolist()
labels = df[label_columns].values.astype(np.float32)

print(f"\nLabel distribution:\n{df[label_columns].sum()}")

# ============================================================================
# 3. TRAIN/EVAL SPLIT
# ============================================================================

train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Evaluation samples: {len(eval_texts)}")

# ============================================================================
# OPTIMIZATION 2: Use faster model
# ============================================================================

if USE_FAST_MODEL:
    model_name = 'distilbert-base-uncased'  # 40% smaller, 60% faster
    print(f"\nUsing FAST model: {model_name}")
else:
    model_name = 'bert-base-uncased'
    print(f"\nUsing standard model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================================
# 4. TOKENIZATION
# ============================================================================

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=128,
        padding='max_length',  # Pad to max_length for faster training
        truncation=True
    )

train_dataset = Dataset.from_dict({
    'text': train_texts,
    'labels': train_labels.tolist()
})

eval_dataset = Dataset.from_dict({
    'text': eval_texts,
    'labels': eval_labels.tolist()
})

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ============================================================================
# 5. MODEL CONFIGURATION
# ============================================================================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    problem_type="multi_label_classification"
)

# ============================================================================
# 6. COMPUTE METRICS
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy()
    binary_predictions = (predictions > 0.5).astype(int)
    
    macro_f1 = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    
    try:
        macro_roc_auc = roc_auc_score(labels, predictions, average='macro')
    except ValueError:
        macro_roc_auc = 0.0
    
    return {
        'macro_f1': macro_f1,
        'macro_roc_auc': macro_roc_auc
    }

# ============================================================================
# OPTIMIZATION 3: Efficient training arguments
# ============================================================================

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,  # Stop early option
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_dir='./logs',
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    dataloader_num_workers=2,  # Parallel data loading
    seed=42,
    report_to="none"
)

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Model: {model_name}")
print(f"Training samples: {len(train_texts)}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Max steps: {MAX_STEPS if MAX_STEPS > 0 else 'No limit'}")
print(f"Batch size: {BATCH_SIZE}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Estimated training time: ", end="")

# Rough time estimation
steps_per_epoch = len(train_texts) // BATCH_SIZE
total_steps = min(steps_per_epoch * NUM_EPOCHS, MAX_STEPS) if MAX_STEPS > 0 else steps_per_epoch * NUM_EPOCHS
seconds_per_step = 0.5 if torch.cuda.is_available() else 2.0  # GPU vs CPU
estimated_minutes = (total_steps * seconds_per_step) / 60
print(f"~{estimated_minutes:.1f} minutes")
print("="*70)

# ============================================================================
# 7. TRAIN
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("\nStarting training...\n")
trainer.train()

# ============================================================================
# 8. EVALUATE
# ============================================================================

print("\n" + "="*70)
print("FINAL EVALUATION RESULTS")
print("="*70)

eval_results = trainer.evaluate()

print(f"\n{'Metric':<30} {'Value':<10}")
print("-" * 45)
print(f"{'Macro-Averaged F1-Score':<30} {eval_results['eval_macro_f1']:.4f}")
print(f"{'Macro-Averaged ROC-AUC':<30} {eval_results['eval_macro_roc_auc']:.4f}")
print(f"{'Evaluation Loss':<30} {eval_results['eval_loss']:.4f}")
print("-" * 45)

# ============================================================================
# 9. SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

predictions = trainer.predict(eval_dataset)
pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
pred_labels = (pred_probs > 0.5).astype(int)

for idx in range(min(3, len(eval_texts))):
    print(f"\nSample {idx + 1}:")
    print(f"Text: {eval_texts[idx][:100]}...")
    print(f"True:      {eval_labels[idx].astype(int)}")
    print(f"Predicted: {pred_labels[idx]}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nModel saved to: ./results")
print(f"Checkpoints saved every 500 steps to: ./results/checkpoint-*")
print(f"\nTo resume from checkpoint if interrupted:")
print(f"trainer.train(resume_from_checkpoint='./results/checkpoint-XXXX')")