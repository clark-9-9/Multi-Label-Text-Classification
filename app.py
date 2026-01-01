from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# ============================================================================
# LOAD MODEL AT STARTUP
# ============================================================================

print("🤖 Loading trained model...")
model_path = os.path.abspath('./Multi Label Text Classification Main/results/checkpoint-200')
print(f"📂 Model path: {model_path}")

# Check if path exists
if not os.path.exists(model_path):
    print(f"❌ Error: Model path does not exist: {model_path}")
    print("Please make sure you have:")
    print("  1. Trained the model (run main2.py)")
    print("  2. The checkpoint-200 folder exists in ./results/")
    exit(1)

model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model.eval()

label_names = [
    'Computer Science',
    'Physics',
    'Mathematics',
    'Statistics',
    'Quantitative Biology',
    'Quantitative Finance'
]

print("✅ Model loaded successfully!")

# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def classify_paper(abstract_text, threshold=0.5):
    """Classify a scientific paper abstract"""
    inputs = tokenizer(
        abstract_text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    predictions = (probabilities >= threshold).astype(int)
    
    results = {
        'predicted_labels': [label_names[i] for i in range(len(predictions)) if predictions[i] == 1],
        'all_probabilities': {label_names[i]: float(probabilities[i]) for i in range(len(label_names))},
        'confidence_scores': {label_names[i]: float(probabilities[i]) for i in range(len(predictions)) if predictions[i] == 1}
    }
    
    return results

# ============================================================================
# HTML TEMPLATE (Embedded)
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scientific Paper Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #a0a0a0;
            font-size: 1.1em;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }

        textarea {
            width: 100%;
            min-height: 250px;
            background: rgba(0, 0, 0, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            color: #e4e4e4;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            transition: all 0.3s ease;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }

        .file-upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }

        .classify-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: all 0.3s ease;
        }

        .classify-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 25px rgba(245, 87, 108, 0.4);
        }

        .label-chip {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 25px;
            margin: 5px;
            font-weight: bold;
            animation: fadeIn 0.5s ease;
        }

        .prob-item {
            margin-bottom: 15px;
        }

        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }

        .prob-bar-container {
            width: 100%;
            height: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            display: flex;
            align-items: center;
            padding-left: 15px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            transition: width 1s ease;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎓 Scientific Paper Classifier</h1>
            <p class="subtitle">AI-Powered Multi-Label Classification</p>
        </header>

        <div class="main-content">
            <div class="card">
                <h2>📝 Input Paper Abstract</h2>
                <textarea id="abstractInput" placeholder="Paste your scientific paper abstract here..."></textarea>
                
                <div style="margin-top: 20px;">
                    <label for="fileInput" class="file-upload-btn" style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; cursor: pointer; transition: all 0.3s ease;">
                        📁 Upload Text File
                    </label>
                    <input type="file" id="fileInput" accept=".txt" style="display: none;" onchange="handleFileUpload(event)">
                    <span id="fileName" style="margin-left: 15px; color: #999; font-size: 14px;"></span>
                </div>
                
                <button class="classify-btn" onclick="classifyPaper()">🚀 Classify Paper</button>
            </div>

            <div class="card">
                <h2>📊 Classification Results</h2>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px;">Analyzing paper...</p>
                </div>

                <div id="results" style="display: none;">
                    <div style="margin-bottom: 30px;">
                        <h3 style="color: #667eea; margin-bottom: 15px;">🎯 Predicted Categories</h3>
                        <div id="predictedLabels"></div>
                    </div>

                    <div>
                        <h3 style="color: #667eea; margin-bottom: 15px;">📈 Confidence Scores</h3>
                        <div id="probabilityBars"></div>
                    </div>
                </div>

                <div id="placeholder" style="text-align: center; padding: 50px 0; color: #666;">
                    <div style="font-size: 4em;">🤖</div>
                    <p>Enter a paper abstract to see results</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('abstractInput').value = e.target.result;
                    document.getElementById('fileName').textContent = `✓ Loaded: ${file.name}`;
                    document.getElementById('fileName').style.color = '#667eea';
                };
                reader.readAsText(file);
            }
        }

        async function classifyPaper() {
            const abstract = document.getElementById('abstractInput').value.trim();
            
            if (!abstract) {
                alert('Please enter a paper abstract!');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').style.display = 'none';
            document.getElementById('placeholder').style.display = 'none';

            try {
                // Call Flask API
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ abstract: abstract })
                });

                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data);
            } catch (error) {
                alert('Error connecting to server: ' + error);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }

        function displayResults(data) {
            document.getElementById('results').style.display = 'block';

            const probabilities = data.all_probabilities;
            const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
            const predicted = sortedProbs.filter(([label, prob]) => prob >= 0.5);

            // Display predicted labels
            const predictedLabelsDiv = document.getElementById('predictedLabels');
            if (predicted.length > 0) {
                predictedLabelsDiv.innerHTML = predicted.map(([label, prob]) => `
                    <div class="label-chip">
                        ✓ ${label}
                        <span style="opacity: 0.9; margin-left: 5px;">${(prob * 100).toFixed(2)}%</span>
                    </div>
                `).join('');
            } else {
                predictedLabelsDiv.innerHTML = '<p style="color: #999;">No categories above 50% threshold</p>';
            }

            // Display probability bars
            const probabilityBarsDiv = document.getElementById('probabilityBars');
            probabilityBarsDiv.innerHTML = sortedProbs.map(([label, prob]) => {
                const percentage = (prob * 100).toFixed(2);
                const barColor = prob >= 0.5 ? 
                    'linear-gradient(90deg, #667eea 0%, #764ba2 100%)' : 
                    'linear-gradient(90deg, #667eea40 0%, #764ba240 100%)';
                
                return `
                    <div class="prob-item">
                        <div class="prob-label">
                            <span>${label}</span>
                            <span style="color: ${prob >= 0.5 ? '#667eea' : '#999'}">${percentage}%</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: ${percentage}%; background: ${barColor}">
                                ${'█'.repeat(Math.floor(prob * 20))}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    </script>
</body>
</html>
'''

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for classification"""
    try:
        data = request.get_json()
        abstract = data.get('abstract', '')
        
        if not abstract:
            return jsonify({'error': 'No abstract provided'}), 400
        
        # Classify the paper
        result = classify_paper(abstract)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': True})

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SCIENTIFIC PAPER CLASSIFIER - WEB APP")
    print("="*70)
    print("\n✅ Model loaded and ready!")
    print("\n🌐 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("\n⚡ Server starting...")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)