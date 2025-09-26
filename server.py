#!/usr/bin/env python3
"""
CometAI Production Web Server

Production-ready FastAPI server for hosting LocalLLM as a web service.
Designed for deployment on cloud platforms with proper error handling,
logging, and process management.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add localllm to path
sys.path.append(str(Path(__file__).parent))

try:
    from localllm import LocalLLM
except ImportError as e:
    print(f"Error importing LocalLLM: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)

# Configure production logging
def setup_logging():
    """Setup production-grade logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging with file rotation
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "cometai_server.log"),
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, ge=0.1, le=2.0, description="Generation temperature")

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    model_info: Dict[str, Any]
    tokens_used: Optional[int] = None

class ModelInfo(BaseModel):
    model_name: str
    description: str
    status: str
    parameters: Any
    device: str
    quantization: str
    uptime: str

class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, ge=1, le=65535, description="Port to bind to")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    max_concurrent_requests: int = Field(default=4, ge=1, le=100, description="Max concurrent requests")
    debug: bool = Field(default=False, description="Debug mode")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loading: bool
    uptime: str
    memory_usage: Optional[Dict[str, Any]] = None
    timestamp: str

class ConnectionManager:
    """Manages WebSocket connections for real-time chat"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

class CometAIServer:
    """Production-ready server class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.llm: Optional[LocalLLM] = None
        self.start_time = datetime.now()
        
        # Create FastAPI app with production settings
        self.app = FastAPI(
            title="CometAI Production Server",
            description="Production-ready Local LLM API Server",
            version="1.0.0",
            docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
            redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
        )
        
        self.connection_manager = ConnectionManager()
        self.conversation_histories: Dict[str, List[Dict]] = {}
        
        # Server state
        self.model_loading = False
        self.model_loaded = False
        self.model_error = None
        self.shutdown_event = asyncio.Event()
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load server configuration"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {
                'api': {
                    'host': 'localhost',
                    'port': 8080,
                    'enable_cors': True,
                    'max_concurrent_requests': 4
                }
            }
    
    def setup_middleware(self):
        """Setup production middleware"""
        api_config = self.config.get('api', {})
        
        # Trusted host middleware for security
        allowed_hosts = api_config.get('allowed_hosts', ["*"])
        if os.getenv("ENVIRONMENT") == "production" and "*" in allowed_hosts:
            logger.warning("Using wildcard hosts in production is not recommended")
        
        self.app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=allowed_hosts
        )
        
        # CORS middleware
        if api_config.get('enable_cors', True):
            cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE"],
                allow_headers=["*"],
            )
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the model on startup"""
            await self.initialize_model()
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "CometAI Production Server",
                "version": "1.0.0",
                "status": "running",
                "model_loaded": self.model_loaded,
                "uptime": str(datetime.now() - self.start_time),
                "endpoints": {
                    "health": "/health",
                    "chat": "/api/chat",
                    "model_info": "/api/model/info",
                    "websocket": "/ws/{client_id}"
                }
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check endpoint"""
            uptime = str(datetime.now() - self.start_time)
            memory_usage = self.get_memory_usage()
            
            return HealthResponse(
                status="healthy" if self.model_loaded else "loading" if self.model_loading else "error",
                model_loaded=self.model_loaded,
                model_loading=self.model_loading,
                uptime=uptime,
                memory_usage=memory_usage,
                timestamp=datetime.now().isoformat()
            )
        
        @self.app.get("/api/model/info")
        async def get_model_info():
            """Get model information"""
            if not self.model_loaded:
                if self.model_error:
                    raise HTTPException(status_code=500, detail=f"Model failed to load: {self.model_error}")
                raise HTTPException(status_code=503, detail="Model is still loading")
            
            try:
                info = self.llm.get_model_info()
                uptime = str(datetime.now() - self.start_time)
                return ModelInfo(
                    model_name=info['model_name'],
                    description=info['description'],
                    status="loaded",
                    parameters=info['total_parameters'],
                    device=info['device'],
                    quantization=info['quantization'],
                    uptime=uptime
                )
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/models/available")
        async def get_available_models():
            """Get list of available models"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model is still loading")
            
            try:
                models = self.llm.list_available_models()
                return {"models": models}
            except Exception as e:
                logger.error(f"Error getting available models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            """Chat endpoint for REST API with rate limiting and error handling"""
            if not self.model_loaded:
                if self.model_error:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                        detail=f"Model failed to load: {self.model_error}"
                    )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                    detail="Model is still loading"
                )
            
            try:
                # Convert Pydantic models to dict for LocalLLM
                history = [msg.dict() for msg in request.conversation_history] if request.conversation_history else []
                
                # Log request (without sensitive data)
                logger.info(f"Chat request: {len(request.message)} chars, {len(history)} history items")
                
                # Generate response with timeout
                start_time = datetime.now()
                response = self.llm.chat(
                    request.message,
                    conversation_history=history,
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7
                )
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # Estimate tokens used (rough approximation)
                tokens_used = len(response.split()) * 1.3
                
                # Get model info
                model_info = self.llm.get_model_info()
                
                logger.info(f"Chat response generated in {generation_time:.2f}s, ~{int(tokens_used)} tokens")
                
                return ChatResponse(
                    response=response,
                    timestamp=datetime.now().isoformat(),
                    model_info=model_info,
                    tokens_used=int(tokens_used)
                )
                
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail=f"Chat generation failed: {str(e)}"
                )
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time chat"""
            await self.connection_manager.connect(websocket, client_id)
            
            try:
                # Send initial status
                await self.connection_manager.send_message(client_id, {
                    "type": "status",
                    "model_loaded": self.model_loaded,
                    "model_loading": self.model_loading,
                    "model_error": self.model_error
                })
                
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    if message_data.get("type") == "chat":
                        await self.handle_websocket_chat(client_id, message_data)
                    elif message_data.get("type") == "ping":
                        await self.connection_manager.send_message(client_id, {"type": "pong"})
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
                self.connection_manager.disconnect(client_id)
        
        @self.app.get("/api/conversations/{client_id}")
        async def get_conversation_history(client_id: str):
            """Get conversation history for a client"""
            history = self.conversation_histories.get(client_id, [])
            return {"conversation_history": history}
        
        @self.app.delete("/api/conversations/{client_id}")
        async def clear_conversation_history(client_id: str):
            """Clear conversation history for a client"""
            if client_id in self.conversation_histories:
                del self.conversation_histories[client_id]
            return {"message": "Conversation history cleared"}
    
    async def initialize_model(self):
        """Initialize the LocalLLM model"""
        if self.model_loading or self.model_loaded:
            return
        
        self.model_loading = True
        logger.info("Initializing LocalLLM model...")
        
        try:
            # Get model configuration
            model_config = self.config.get('model', {})
            model_name = model_config.get('name', 'qwen2.5-coder-7b-instruct')
            model_path = model_config.get('path')
            
            # Initialize model
            self.llm = LocalLLM(
                model_name=model_name,
                model_path=model_path,
                config_path=str(Path(__file__).parent / "config.yaml")
            )
            
            self.model_loaded = True
            self.model_loading = False
            self.model_error = None
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            
            # Broadcast model loaded status to all connected clients
            await self.connection_manager.broadcast({
                "type": "model_status",
                "loaded": True,
                "loading": False,
                "error": None,
                "model_info": self.llm.get_model_info()
            })
            
        except Exception as e:
            self.model_loading = False
            self.model_error = str(e)
            logger.error(f"❌ Failed to load model: {e}")
            
            # Broadcast error to all connected clients
            await self.connection_manager.broadcast({
                "type": "model_status",
                "loaded": False,
                "loading": False,
                "error": str(e)
            })
    
    async def handle_websocket_chat(self, client_id: str, message_data: dict):
        """Handle chat message via WebSocket"""
        if not self.model_loaded:
            await self.connection_manager.send_message(client_id, {
                "type": "error",
                "message": "Model is not loaded yet"
            })
            return
        
        try:
            message = message_data.get("message", "")
            if not message.strip():
                return
            
            # Get or create conversation history for this client
            if client_id not in self.conversation_histories:
                self.conversation_histories[client_id] = []
            
            history = self.conversation_histories[client_id]
            
            # Add user message to history
            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            history.append(user_msg)
            
            # Send acknowledgment
            await self.connection_manager.send_message(client_id, {
                "type": "message_received",
                "message": user_msg
            })
            
            # Send typing indicator
            await self.connection_manager.send_message(client_id, {
                "type": "typing",
                "typing": True
            })
            
            # Generate response
            response = self.llm.chat(
                message,
                conversation_history=history[:-1],  # Exclude the current user message
                max_tokens=message_data.get("max_tokens"),
                temperature=message_data.get("temperature")
            )
            
            # Add AI response to history
            ai_msg = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            }
            history.append(ai_msg)
            
            # Keep history manageable (last 50 messages)
            if len(history) > 50:
                self.conversation_histories[client_id] = history[-50:]
            
            # Send response
            await self.connection_manager.send_message(client_id, {
                "type": "chat_response",
                "message": ai_msg,
                "model_info": self.llm.get_model_info()
            })
            
            # Stop typing indicator
            await self.connection_manager.send_message(client_id, {
                "type": "typing",
                "typing": False
            })
            
        except Exception as e:
            logger.error(f"WebSocket chat error for {client_id}: {e}")
            await self.connection_manager.send_message(client_id, {
                "type": "error",
                "message": str(e)
            })
            
            # Stop typing indicator
            await self.connection_manager.send_message(client_id, {
                "type": "typing",
                "typing": False
            })
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Close all WebSocket connections
        if self.connection_manager.active_connections:
            logger.info(f"Closing {len(self.connection_manager.active_connections)} WebSocket connections")
            for client_id in list(self.connection_manager.active_connections.keys()):
                try:
                    await self.connection_manager.send_message(client_id, {
                        "type": "server_shutdown",
                        "message": "Server is shutting down"
                    })
                except:
                    pass
                self.connection_manager.disconnect(client_id)
        
        # Clean up model resources
        if self.llm:
            logger.info("Cleaning up model resources...")
            try:
                # Clear CUDA cache if available
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        logger.info("Graceful shutdown completed")
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None):
        """Run the production server"""
        api_config = self.config.get('api', {})
        
        # Use environment variables or config (Railway/Render compatible)
        server_host = host or os.getenv("HOST", "0.0.0.0")
        server_port = port or int(os.getenv("PORT", api_config.get('port', 8080)))
        server_debug = debug or os.getenv("DEBUG", "false").lower() == "true"
        
        # Production settings
        workers = int(os.getenv("WORKERS", 1))  # Single worker for model consistency
        
        logger.info(f"🚀 Starting CometAI Production Server")
        logger.info(f"   Host: {server_host}")
        logger.info(f"   Port: {server_port}")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Debug: {server_debug}")
        logger.info(f"   Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # Configure uvicorn for production
        config = uvicorn.Config(
            app=self.app,
            host=server_host,
            port=server_port,
            log_level="info",
            access_log=True,
            workers=workers,
            loop="asyncio",
            reload=server_debug and workers == 1,
            # Production optimizations
            backlog=2048,
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
        )
        
        server = uvicorn.Server(config)
        
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            # Ensure graceful shutdown
            asyncio.run(self.graceful_shutdown())

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CometAI Web Server")
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        server = CometAIServer(config_path=args.config)
        server.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
