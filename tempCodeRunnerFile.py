HTML_TEMPLATE='''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Scientific Paper Classifier</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }

#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#             color: #e4e4e4;
#             min-height: 100vh;
#             padding: 20px;
#         }

#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#         }

#         header {
#             text-align: center;
#             margin-bottom: 40px;
#             padding: 30px 20px;
#             background: rgba(255, 255, 255, 0.05);
#             border-radius: 20px;
#             backdrop-filter: blur(10px);
#             border: 1px solid rgba(255, 255, 255, 0.1);
#         }

#         h1 {
#             font-size: 2.5em;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#             margin-bottom: 10px;
#         }

#         .subtitle {
#             color: #a0a0a0;
#             font-size: 1.1em;
#         }

#         .main-content {
#             display: grid;
#             grid-template-columns: 1fr 1fr;
#             gap: 30px;
#             margin-bottom: 30px;
#         }

#         @media (max-width: 968px) {
#             .main-content {
#                 grid-template-columns: 1fr;
#             }
#         }

#         .card {
#             background: rgba(255, 255, 255, 0.05);
#             border-radius: 20px;
#             padding: 30px;
#             backdrop-filter: blur(10px);
#             border: 1px solid rgba(255, 255, 255, 0.1);
#             box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#         }

#         .card h2 {
#             color: #667eea;
#             margin-bottom: 20px;
#             font-size: 1.5em;
#             display: flex;
#             align-items: center;
#             gap: 10px;
#         }

#         textarea {
#             width: 100%;
#             min-height: 250px;
#             background: rgba(0, 0, 0, 0.3);
#             border: 2px solid rgba(255, 255, 255, 0.1);
#             border-radius: 12px;
#             padding: 15px;
#             color: #e4e4e4;
#             font-size: 14px;
#             line-height: 1.6;
#             resize: vertical;
#             transition: all 0.3s ease;
#         }

#         textarea:focus {
#             outline: none;
#             border-color: #667eea;
#             box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
#         }

#         .file-upload {
#             margin-top: 20px;
#         }

#         .file-upload-btn {
#             display: inline-block;
#             padding: 12px 24px;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border-radius: 10px;
#             cursor: pointer;
#             transition: all 0.3s ease;
#             border: none;
#             font-size: 16px;
#         }

#         .file-upload-btn:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
#         }

#         input[type="file"] {
#             display: none;
#         }

#         .classify-btn {
#             width: 100%;
#             padding: 15px;
#             background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#             color: white;
#             border: none;
#             border-radius: 12px;
#             font-size: 18px;
#             font-weight: bold;
#             cursor: pointer;
#             margin-top: 20px;
#             transition: all 0.3s ease;
#         }

#         .classify-btn:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 5px 25px rgba(245, 87, 108, 0.4);
#         }

#         .classify-btn:active {
#             transform: translateY(0);
#         }

#         .results-section {
#             display: none;
#         }

#         .results-section.active {
#             display: block;
#         }

#         .predicted-labels {
#             margin-bottom: 30px;
#         }

#         .label-chip {
#             display: inline-block;
#             padding: 10px 20px;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             border-radius: 25px;
#             margin: 5px;
#             font-weight: bold;
#             animation: fadeIn 0.5s ease;
#         }

#         .confidence-text {
#             font-size: 0.9em;
#             opacity: 0.9;
#             margin-left: 5px;
#         }

#         .probability-bars {
#             margin-top: 20px;
#         }

#         .prob-item {
#             margin-bottom: 15px;
#             animation: slideIn 0.5s ease;
#         }

#         .prob-label {
#             display: flex;
#             justify-content: space-between;
#             margin-bottom: 5px;
#             font-size: 14px;
#         }

#         .prob-bar-container {
#             width: 100%;
#             height: 30px;
#             background: rgba(0, 0, 0, 0.3);
#             border-radius: 15px;
#             overflow: hidden;
#             position: relative;
#         }

#         .prob-bar {
#             height: 100%;
#             background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#             border-radius: 15px;
#             display: flex;
#             align-items: center;
#             justify-content: flex-end;
#             padding-right: 15px;
#             color: white;
#             font-weight: bold;
#             font-size: 12px;
#             transition: width 1s ease;
#         }

#         .icon {
#             font-size: 1.2em;
#         }

#         @keyframes fadeIn {
#             from {
#                 opacity: 0;
#                 transform: scale(0.8);
#             }
#             to {
#                 opacity: 1;
#                 transform: scale(1);
#             }
#         }

#         @keyframes slideIn {
#             from {
#                 opacity: 0;
#                 transform: translateX(-20px);
#             }
#             to {
#                 opacity: 1;
#                 transform: translateX(0);
#             }
#         }

#         .loading {
#             display: none;
#             text-align: center;
#             margin: 20px 0;
#         }

#         .loading.active {
#             display: block;
#         }

#         .spinner {
#             border: 4px solid rgba(255, 255, 255, 0.1);
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 50px;
#             height: 50px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto;
#         }

#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }

#         .example-btn {
#             padding: 8px 16px;
#             background: rgba(102, 126, 234, 0.2);
#             border: 1px solid #667eea;
#             color: #667eea;
#             border-radius: 8px;
#             cursor: pointer;
#             font-size: 14px;
#             margin-top: 10px;
#             transition: all 0.3s ease;
#         }

#         .example-btn:hover {
#             background: rgba(102, 126, 234, 0.3);
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <header>
#             <h1>🎓 Scientific Paper Classifier</h1>
#             <p class="subtitle">AI-Powered Multi-Label Classification using Deep Learning</p>
#         </header>

#         <div class="main-content">
#             <!-- Input Section -->
#             <div class="card">
#                 <h2><span class="icon">📝</span> Input Paper Abstract</h2>
#                 <textarea id="abstractInput" placeholder="Paste your scientific paper abstract here...

# Example: 'Deep learning has revolutionized computer vision through convolutional neural networks. We present a novel architecture that combines attention mechanisms with residual connections...'"></textarea>
                
#                 <button class="example-btn" onclick="loadExample()">Load Example Paper</button>
                
#                 <div class="file-upload">
#                     <label for="fileInput" class="file-upload-btn">
#                         📁 Or Upload Text File
#                     </label>
#                     <input type="file" id="fileInput" accept=".txt" onchange="handleFileUpload(event)">
#                 </div>

#                 <button class="classify-btn" onclick="classifyPaper()">
#                     🚀 Classify Paper
#                 </button>
#             </div>

#             <!-- Results Section -->
#             <div class="card">
#                 <h2><span class="icon">📊</span> Classification Results</h2>
                
#                 <div class="loading" id="loading">
#                     <div class="spinner"></div>
#                     <p style="margin-top: 15px;">Analyzing paper...</p>
#                 </div>

#                 <div class="results-section" id="results">
#                     <div class="predicted-labels">
#                         <h3 style="color: #667eea; margin-bottom: 15px;">🎯 Predicted Categories</h3>
#                         <div id="predictedLabels"></div>
#                     </div>

#                     <div class="probability-bars">
#                         <h3 style="color: #667eea; margin-bottom: 15px;">📈 Confidence Scores</h3>
#                         <div id="probabilityBars"></div>
#                     </div>
#                 </div>

#                 <div id="placeholder" style="text-align: center; padding: 50px 0; color: #666;">
#                     <div style="font-size: 4em; margin-bottom: 20px;">🤖</div>
#                     <p>Enter a paper abstract and click "Classify Paper" to see results</p>
#                 </div>
#             </div>
#         </div>
#     </div>

#     <script>
#         // Example papers for demo
#         const examplePaper = `Deep learning has revolutionized computer vision through convolutional neural networks. We present a novel architecture that combines attention mechanisms with residual connections to improve image classification accuracy. Our method achieves state-of-the-art results on ImageNet with fewer parameters than existing approaches. We also provide mathematical proofs of convergence.`;

#         // Label names (must match your model)
#         const labelNames = [
#             'Computer Science',
#             'Physics',
#             'Mathematics',
#             'Statistics',
#             'Quantitative Biology',
#             'Quantitative Finance'
#         ];

#         function loadExample() {
#             document.getElementById('abstractInput').value = examplePaper;
#         }

#         function handleFileUpload(event) {
#             const file = event.target.files[0];
#             if (file) {
#                 const reader = new FileReader();
#                 reader.onload = function(e) {
#                     document.getElementById('abstractInput').value = e.target.result;
#                 };
#                 reader.readAsText(file);
#             }
#         }

#         function classifyPaper() {
#             const abstract = document.getElementById('abstractInput').value.trim();
            
#             if (!abstract) {
#                 alert('Please enter a paper abstract or upload a file!');
#                 return;
#             }

#             // Show loading
#             document.getElementById('loading').classList.add('active');
#             document.getElementById('results').classList.remove('active');
#             document.getElementById('placeholder').style.display = 'none';

#             // Simulate classification (replace with actual model API call)
#             setTimeout(() => {
#                 // MOCK RESULTS - Replace this with actual model prediction
#                 const mockResults = simulateClassification(abstract);
#                 displayResults(mockResults);
#             }, 1500);
#         }

#         function simulateClassification(text) {
#             // This simulates your Python model output
#             // REPLACE THIS with actual API call to your Python backend
            
#             // Simple keyword-based simulation for demo
#             const keywords = text.toLowerCase();
#             let probabilities = {};
            
#             // Mock probability calculation based on keywords
#             probabilities['Computer Science'] = keywords.includes('learning') || keywords.includes('neural') || keywords.includes('algorithm') ? 0.7 + Math.random() * 0.2 : Math.random() * 0.3;
#             probabilities['Mathematics'] = keywords.includes('proof') || keywords.includes('mathematical') || keywords.includes('theorem') ? 0.6 + Math.random() * 0.2 : Math.random() * 0.3;
#             probabilities['Physics'] = keywords.includes('quantum') || keywords.includes('particle') || keywords.includes('energy') ? 0.6 + Math.random() * 0.3 : Math.random() * 0.25;
#             probabilities['Statistics'] = keywords.includes('statistical') || keywords.includes('data') || keywords.includes('analysis') ? 0.5 + Math.random() * 0.3 : Math.random() * 0.3;
#             probabilities['Quantitative Biology'] = keywords.includes('biological') || keywords.includes('protein') || keywords.includes('gene') ? 0.5 + Math.random() * 0.3 : Math.random() * 0.15;
#             probabilities['Quantitative Finance'] = keywords.includes('financial') || keywords.includes('market') || keywords.includes('trading') ? 0.5 + Math.random() * 0.3 : Math.random() * 0.2;

#             return probabilities;
#         }

#         function displayResults(probabilities) {
#             // Hide loading
#             document.getElementById('loading').classList.remove('active');
#             document.getElementById('results').classList.add('active');

#             // Sort probabilities
#             const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

#             // Get predicted labels (threshold 0.5)
#             const predicted = sortedProbs.filter(([label, prob]) => prob >= 0.5);

#             // Display predicted labels
#             const predictedLabelsDiv = document.getElementById('predictedLabels');
#             if (predicted.length > 0) {
#                 predictedLabelsDiv.innerHTML = predicted.map(([label, prob]) => `
#                     <div class="label-chip">
#                         ✓ ${label}
#                         <span class="confidence-text">${(prob * 100).toFixed(2)}%</span>
#                     </div>
#                 `).join('');
#             } else {
#                 predictedLabelsDiv.innerHTML = '<p style="color: #999;">No categories above 50% confidence threshold</p>';
#             }

#             // Display probability bars
#             const probabilityBarsDiv = document.getElementById('probabilityBars');
#             probabilityBarsDiv.innerHTML = sortedProbs.map(([label, prob], index) => {
#                 const percentage = (prob * 100).toFixed(2);
#                 const barColor = prob >= 0.5 ? 
#                     'linear-gradient(90deg, #667eea 0%, #764ba2 100%)' : 
#                     'linear-gradient(90deg, #667eea40 0%, #764ba240 100%)';
                
#                 return `
#                     <div class="prob-item" style="animation-delay: ${index * 0.1}s">
#                         <div class="prob-label">
#                             <span>${label}</span>
#                             <span style="color: ${prob >= 0.5 ? '#667eea' : '#999'}">${percentage}%</span>
#                         </div>
#                         <div class="prob-bar-container">
#                             <div class="prob-bar" style="width: ${percentage}%; background: ${barColor}">
#                                 ${'█'.repeat(Math.floor(prob * 20))}
#                             </div>
#                         </div>
#                     </div>
#                 `;
#             }).join('');
#         }

#         // Auto-load example on page load (optional)
#         // window.onload = () => loadExample();
#     </script>
# </body>
# </html>
# '''
