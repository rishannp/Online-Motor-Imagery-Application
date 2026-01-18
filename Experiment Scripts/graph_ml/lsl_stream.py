# lsl_stream.py
from pylsl import resolve_stream, StreamInlet
from config import LSL_STREAM_TYPE, LSL_TIMEOUT_SEC

def get_inlet():
    streams = resolve_stream('type', LSL_STREAM_TYPE, timeout=LSL_TIMEOUT_SEC)
    if not streams:
        raise RuntimeError(f"No LSL stream found for type='{LSL_STREAM_TYPE}' within {LSL_TIMEOUT_SEC}s.")
    return StreamInlet(streams[0], max_buflen=5)
