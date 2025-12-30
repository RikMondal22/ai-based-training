import uvicorn
import os
from app.main import app

if __name__ == "__main__":
    # Render assigns PORT environment variable
    port = int(os.getenv("PORT", 10000))
    
    # CRITICAL: Must bind to 0.0.0.0 and the PORT Render assigns
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=port,
        log_level="info"
    )
