from flask import Flask, jsonify, request, render_template
import torch
import os
import json
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from scipy.stats import pearsonr
from dataloader import DataProcessor

app = Flask(__name__)

class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict_similarity(self, sentence1, sentence2):
        # Tokenize input sentences
        encoded_input = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            similarity_score = torch.sigmoid(logits).item()  # Assuming binary classification
        
        return similarity_score

# Load best model from JSON file
with open("best_model_paths.json", "r") as f:
    best_model_path = json.load(f)
# best_model_path="C:/Users/panka/Desktop/semantic_textual_relatedness/Semantic_Relatedness_SemEval2024/all_codes_STR/all_codes_STR/saved_models_train_py_format/eng/epoch_1"
best_model_path="pankaj100567/semantic_textual_relatedness"
# best_model_path="Semantic_Relatedness_SemEval2024/all_codes_STR/all_codes_STR/saved_models_train_py_format/eng/epoch_1"
# Load model and tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained(best_model_path)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Predictor instance
predictor = Predictor(model, tokenizer, device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input sentences from the form
    sentence1 = request.form['sentence1']
    sentence2 = request.form['sentence2']
    
    # Make prediction
    similarity_score = predictor.predict_similarity(sentence1, sentence2)
    
    # Return the similarity score
    return render_template('result.html', sentence1=sentence1, sentence2=sentence2, similarity_score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
