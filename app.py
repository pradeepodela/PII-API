# app.py
import os
import json
import logging
from typing import Dict, List, Union
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from gliner import GLiNER

# API Configuration
PORT=5000
FLASK_ENV='production'

# Optional: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL='INFO'

# Optional: Set maximum request size (in MB)
MAX_CONTENT_LENGTH=10

# Optional: Set model name
MODEL_NAME='urchade/gliner_multi_pii-v1'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pii_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load GLiNER model
try:
    MODEL = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    logger.info("GLiNER model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load GLiNER model: {str(e)}")
    raise

# Default PII labels
DEFAULT_LABELS = [
    "person", "organization", "address", "email", "phone number", 
    "social security number", "credit card number", "passport number", 
    "driver license", "bank account number", "date of birth", 
    "medical record number", "insurance policy number", "property registration number",
    "employee ID number", "tax ID number", "full address", "personally identifiable information"
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API status"""
    return jsonify({"status": "healthy", "model": "GLiNER PII Extractor"}), 200

@app.route('/api/extract', methods=['POST'])
def extract_pii():
    """
    Extract PII entities from provided text
    
    Request body:
    {
        "text": "Text containing PII to analyze",
        "labels": "comma,separated,labels" (optional),
        "threshold": 0.5 (optional),
        "nested_ner": false (optional)
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get required parameters
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get optional parameters with defaults
        labels_str = data.get('labels')
        threshold = float(data.get('threshold', 0.5))
        nested_ner = bool(data.get('nested_ner', False))
        
        # Process labels
        if labels_str:
            labels = [label.strip() for label in labels_str.split(',')]
        else:
            labels = DEFAULT_LABELS
            
        # Validate threshold
        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
            
        # Log request details
        logger.info(f"Processing PII extraction request: {len(text)} chars, {len(labels)} labels, threshold={threshold}")
        
        # Extract entities
        entities = MODEL.predict_entities(
            text, 
            labels, 
            flat_ner=not nested_ner, 
            threshold=threshold
        )
        
        # Format results
        results = {
            "text": text,
            "entities": [
                {
                    "entity": entity["label"],
                    "word": entity["text"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": float(entity.get("score", 0)),
                }
                for entity in entities
            ],
        }
        
        logger.info(f"PII extraction completed: {len(results['entities'])} entities found")
        
        return jsonify(results), 200
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

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

