import os
import json
import logging
import requests
import time
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
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
HF_MODEL = "urchade/gliner_multi_pii-v1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Default PII labels
DEFAULT_LABELS = [
    "person", "organization", "address", "email", "phone number", 
    "social security number", "credit card number", "passport number", 
    "driver license", "bank account number", "date of birth"
]

def query_huggingface_ner(text: str, labels: List[str], threshold: float = 0.5, max_retries: int = 3):
    """
    Query Hugging Face for Named Entity Recognition
    GLiNER models work differently than standard HF NER models
    """
    if not HF_API_TOKEN:
        raise Exception("HF_API_TOKEN environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # GLiNER expects a different payload format
    payload = {
        "inputs": text,
        "parameters": {
            "entities": labels,  # GLiNER uses 'entities' not 'labels'
            "threshold": threshold
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting HF API call (attempt {attempt + 1}/{max_retries})")
            
            response = requests.post(
                HF_API_URL, 
                headers=headers, 
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            logger.info(f"HF API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"HF API Success: {len(result) if isinstance(result, list) else 'Unknown'} entities")
                return result
                
            elif response.status_code == 503:
                # Model is loading, wait and retry
                logger.warning(f"Model loading, waiting 10 seconds... (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                    
            elif response.status_code == 429:
                # Rate limited
                logger.warning(f"Rate limited, waiting 5 seconds... (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
            else:
                logger.error(f"HF API Error {response.status_code}: {response.text}")
                raise Exception(f"HF API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception("Request timeout after multiple attempts")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception(f"Request failed: {str(e)}")
    
    raise Exception("Max retries exceeded")

def fallback_simple_pii_detection(text: str) -> List[Dict]:
    """
    Simple fallback PII detection using regex patterns
    """
    import re
    
    entities = []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append({
            "entity": "email",
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "score": 0.9
        })
    
    # Phone pattern (simple US format)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    for match in re.finditer(phone_pattern, text):
        entities.append({
            "entity": "phone number",
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "score": 0.8
        })
    
    # SSN pattern
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    for match in re.finditer(ssn_pattern, text):
        entities.append({
            "entity": "social security number",
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
            "score": 0.95
        })
    
    return entities

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "GLiNER PII Extractor (HF API)",
        "hf_token_configured": bool(HF_API_TOKEN)
    }), 200

@app.route('/api/extract', methods=['POST'])
def extract_pii():
    """Extract PII entities from provided text"""
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) > 10000:  # Limit text size
            return jsonify({"error": "Text too long (max 10,000 characters)"}), 400
        
        # Get optional parameters
        labels_str = data.get('labels')
        threshold = float(data.get('threshold', 0.5))
        use_fallback = bool(data.get('use_fallback', True))
        
        # Process labels
        if labels_str:
            labels = [label.strip() for label in labels_str.split(',')]
        else:
            labels = DEFAULT_LABELS
            
        # Validate threshold
        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
            
        logger.info(f"Processing PII extraction: {len(text)} chars, {len(labels)} labels")
        
        entities = []
        method_used = "unknown"
        
        try:
            # Try Hugging Face API first
            hf_result = query_huggingface_ner(text, labels, threshold)
            method_used = "huggingface_api"
            
            # Handle different response formats from HF
            if isinstance(hf_result, list):
                entities = [
                    {
                        "entity": entity.get("entity_group", entity.get("label", "unknown")),
                        "word": entity.get("word", entity.get("text", "")),
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0),
                        "score": float(entity.get("score", 0)),
                    }
                    for entity in hf_result
                    if entity.get("score", 0) >= threshold
                ]
            else:
                logger.warning(f"Unexpected HF response format: {type(hf_result)}")
                entities = []
                
        except Exception as hf_error:
            logger.error(f"Hugging Face API failed: {str(hf_error)}")
            
            if use_fallback:
                logger.info("Using fallback regex-based detection")
                entities = fallback_simple_pii_detection(text)
                method_used = "fallback_regex"
            else:
                raise hf_error
        
        # Format results
        results = {
            "text": text,
            "entities": entities,
            "method_used": method_used,
            "processing_time": round(time.time() - start_time, 2),
            "entity_count": len(entities)
        }
        
        logger.info(f"PII extraction completed: {len(entities)} entities found using {method_used} in {results['processing_time']}s")
        
        return jsonify(results), 200
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"Error processing request: {str(e)} (took {processing_time}s)")
        return jsonify({
            "error": str(e),
            "processing_time": processing_time
        }), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample data"""
    sample_text = "John Doe lives at 123 Main St. His email is john@example.com and phone is 555-123-4567."
    
    try:
        # Test the extraction
        test_data = {
            "text": sample_text,
            "threshold": 0.5,
            "use_fallback": True
        }
        
        # Simulate the extraction process
        entities = fallback_simple_pii_detection(sample_text)
        
        return jsonify({
            "status": "test_successful",
            "sample_text": sample_text,
            "entities_found": len(entities),
            "entities": entities
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "test_failed",
            "error": str(e)
        }), 500

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
