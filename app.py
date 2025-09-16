import cv2
import asyncio
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import threading
import queue
import time

app = FastAPI(title="YOLO Video Streaming API")

# Global variables
frame_queue = queue.Queue(maxsize=10)
model = None
streaming_active = False

def initialize_yolo():
    """Initialize YOLO model"""
    global model
    model = YOLO("runs/detect/train/weights/best.pt")

def video_capture_thread():
    """Background thread to capture and process video frames"""
    global streaming_active

    while streaming_active:
        try:
            # Process frames from TCP stream
            for res in model("tcp://127.0.0.1:8888", stream=True):
                if not streaming_active:
                    break

                # Get annotated frame
                annotated_frame = res.plot()

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                # Add to queue (non-blocking)
                try:
                    frame_queue.put(frame_bytes, block=False)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(frame_bytes, block=False)
                    except queue.Empty:
                        pass

        except Exception as e:
            print(f"Error in video capture: {e}")
            time.sleep(1)  # Wait before retrying

def generate_frames():
    """Generator function to yield video frames"""
    while streaming_active:
        try:
            # Get frame from queue with timeout
            frame_bytes = frame_queue.get(timeout=1.0)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except queue.Empty:
            # No frame available, yield empty frame or continue
            continue
        except Exception as e:
            print(f"Error generating frame: {e}")
            break

@app.on_event("startup")
async def startup_event():
    """Initialize YOLO model on startup"""
    initialize_yolo()

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve HTML page with video stream"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Video Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
            }
            h1 {
                color: #333;
            }
            .video-container {
                margin: 20px 0;
                display: inline-block;
                border: 2px solid #333;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            img {
                display: block;
                max-width: 100%;
                height: auto;
            }
            .controls {
                margin: 20px 0;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 0 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .status {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
            }
            .status.streaming {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.stopped {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO Video Stream</h1>
            <div id="status" class="status stopped">Stream Stopped</div>

            <div class="controls">
                <button onclick="startStream()" id="startBtn">Start Stream</button>
                <button onclick="stopStream()" id="stopBtn" disabled>Stop Stream</button>
            </div>

            <div class="video-container">
                <img id="videoFeed" src="/static/placeholder.jpg" alt="Video Stream"
                     style="width: 800px; height: 600px; background-color: #ddd;">
            </div>
        </div>

        <script>
            let streamActive = false;

            function startStream() {
                fetch('/start_stream', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'started') {
                            streamActive = true;
                            document.getElementById('videoFeed').src = '/video_feed';
                            document.getElementById('startBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = false;
                            document.getElementById('status').className = 'status streaming';
                            document.getElementById('status').textContent = 'Stream Active';
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }

            function stopStream() {
                fetch('/stop_stream', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'stopped') {
                            streamActive = false;
                            document.getElementById('videoFeed').src = '/static/placeholder.jpg';
                            document.getElementById('startBtn').disabled = false;
                            document.getElementById('stopBtn').disabled = true;
                            document.getElementById('status').className = 'status stopped';
                            document.getElementById('status').textContent = 'Stream Stopped';
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }

            // Handle image load errors
            document.getElementById('videoFeed').onerror = function() {
                if (streamActive) {
                    // Retry loading the image after a short delay
                    setTimeout(() => {
                        this.src = '/video_feed?' + new Date().getTime();
                    }, 1000);
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/start_stream")
async def start_stream():
    """Start the video stream"""
    global streaming_active

    if not streaming_active:
        streaming_active = True
        # Start video capture in background thread
        thread = threading.Thread(target=video_capture_thread, daemon=True)
        thread.start()

        return {"status": "started", "message": "Video stream started"}
    else:
        return {"status": "already_running", "message": "Stream is already active"}

@app.post("/stop_stream")
async def stop_stream():
    """Stop the video stream"""
    global streaming_active

    streaming_active = False

    # Clear the frame queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    return {"status": "stopped", "message": "Video stream stopped"}

@app.get("/video_feed")
async def video_feed():
    """Stream video frames"""
    if not streaming_active:
        return Response("Stream not active", status_code=404)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
async def get_status():
    """Get current streaming status"""
    return {
        "streaming": streaming_active,
        "queue_size": frame_queue.qsize(),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
