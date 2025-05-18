from app import app
from waitress import serve
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = 5000
    logger.info(f"Starting production server on port {port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
