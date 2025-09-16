#!/usr/bin/env python3
"""
Startup script for YOLO FastAPI video streaming server
"""
import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting YOLO Video Streaming Server...")
    print("Server will be available at: http://0.0.0.0:8000")
    print("Press Ctrl+C to stop the server")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )
