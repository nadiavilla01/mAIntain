import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


try:
    from dotenv import load_dotenv
    ENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)
except Exception as e:
    print(f"[BOOT] .env load error: {e}")

print(f"[BOOT] OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")


from routes import machines, insights, ai_chat, history, fault_api
from routes.fault_api import STATIC_DIR as FAULT_STATIC_DIR

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=FAULT_STATIC_DIR), name="static")


app.include_router(machines.router, prefix="/machines")
app.include_router(insights.router, prefix="/insights")
app.include_router(ai_chat.router, prefix="/api/ai-chat")
app.include_router(history.router, prefix="/api/history")
app.include_router(fault_api.router, prefix="/fault-detection")


@app.get("/debug/llm-status")
def debug_llm_status():
    return {"has_openai_key": bool(os.getenv("OPENAI_API_KEY"))}