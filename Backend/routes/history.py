from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("")
async def get_history():
    return [
        {
            "timestamp": "21 Jul, 19:38",
            "machine": "Chiller",
            "type": "Status Change",
            "detail": "Chiller status changed to 'Normal'",
        },
        {
            "timestamp": "21 Jul, 18:38",
            "machine": "Pump A",
            "type": "AI Suggestion",
            "detail": "AI suggested maintenance for Pump A",
        },
        {
            "timestamp": "21 Jul, 17:38",
            "machine": "Robot Arm",
            "type": "Status Change",
            "detail": "Robot Arm status changed to 'Critical'",
        },
        {
            "timestamp": "21 Jul, 17:38",
            "machine": "Mixer",
            "type": "Anomaly Detected",
            "detail": "Mixer anomaly: Temp Spike",
        },
        {
            "timestamp": "21 Jul, 13:38",
            "machine": "Conveyor 1",
            "type": "Report Generated",
            "detail": "Report generated for Conveyor 1",
        },
    ]