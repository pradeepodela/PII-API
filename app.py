import os
import json
import logging
import requests
from typing import Dict, List, Union
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Hugging Face configuration
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')  # Set this in Vercel environment variables
HF_MODEL = "urchade/gliner_multi_pii-v1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Default PII labels
DEFAULT_LABELS = [
    "person", "organization", "address", "email", "phone number", 
    "social security number", "credit card number", "passport number", 
    "driver license", "bank account number", "date of birth", 
    "medical record number", "insurance policy number", "property registration number",
    "employee ID number", "tax ID number", "full address", "personally identifiable information"
]

def query_huggingface(text: str, labels: List[str], threshold: float = 0.5):
    """Query Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    payload = {
        "inputs": text,
        "parameters": {
            "labels": labels,
            "threshold": threshold
        }
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"HF API Error: {response.status_code} - {response.text}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "GLiNER PII Extractor (HF API)"}), 200

@app.route('/api/extract', methods=['POST'])
def extract_pii():
    """Extract PII entities from provided text using Hugging Face API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        labels_str = data.get('labels')
        threshold = float(data.get('threshold', 0.5))
        
        if labels_str:
            labels = [label.strip() for label in labels_str.split(',')]
        else:
            labels = DEFAULT_LABELS
            
        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
            
        logger.info(f"Processing PII extraction: {len(text)} chars, {len(labels)} labels")
        
        # Query Hugging Face API
        hf_result = query_huggingface(text, labels, threshold)
        
        # Format results to match original API
        results = {
            "text": text,
            "entities": [
                {
                    "entity": entity.get("entity_group", entity.get("label", "")),
                    "word": entity.get("word", ""),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "score": float(entity.get("score", 0)),
                }
                for entity in hf_result
            ],
        }
        
        logger.info(f"PII extraction completed: {len(results['entities'])} entities found")
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500
