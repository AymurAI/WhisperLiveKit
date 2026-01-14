import asyncio
import websockets
import aiofiles
import numpy as np
import librosa
import json
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


async def send_audio(
    websocket: websockets.ClientConnection,
    source: str | Path,
    chunk_size_s: float = 1.0,
    simu_realtime: bool = False,
) -> float:
    """Send audio chunks to websocket"""

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    audio, sr = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sr
    chunk_size = int(sr * chunk_size_s)
    logger.info(
        f"Streaming {path.name} ({duration:.2f}s, sr={sr}, chunk={chunk_size_s:.2f}s){' [Simu-RT]' if simu_realtime else ''}"
    )

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) == 0:
            continue
        await _send_chunk(websocket, chunk)
        if simu_realtime:
            await asyncio.sleep(chunk_size_s)

    await websocket.send(b"")  # End of stream
    return duration


async def _send_chunk(websocket: websockets.ClientConnection, chunk: np.ndarray):
    """Encode PCM chunk to bytes and send"""
    chunk_int16 = (chunk * 32768).astype(np.int16)
    await websocket.send(chunk_int16.tobytes())


async def receive_updates(
    websocket: websockets.ClientConnection,
    print_raw_json: bool = False,
):
    """Receive server responses and return the last active transcription message."""
    last_active_transcription = None
    while True:
        try:
            msg = await websocket.recv()
            resp = json.loads(msg)
            if resp.get("status") == "active_transcription":
                last_active_transcription = resp
            if print_raw_json:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
            if resp.get("type") == "ready_to_stop":
                break

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break
    return last_active_transcription


async def run_client(
    source,
    uri="ws://localhost:8000/asr",
    chunk_size: float = 1.0,
    simu_realtime: bool = False,
    output_json: str | None = None,
    print_raw_json: bool = False,
):
    """Main pipeline: send audio and save last active transcription message."""
    async with websockets.connect(uri) as ws:
        logger.info(f"Connected to {uri}")
        recv_task = asyncio.create_task(
            receive_updates(ws, print_raw_json=print_raw_json)
        )

        await send_audio(
            ws, source, chunk_size_s=chunk_size, simu_realtime=simu_realtime
        )
        last_active_transcription = await recv_task

        # Optionally write the last active transcription message to JSON file.
        if output_json:
            try:
                out_path = Path(output_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                    await f.write(
                        json.dumps(
                            last_active_transcription, ensure_ascii=False, indent=2
                        )
                    )
                logger.info(f"Saved active transcription JSON to {out_path}")
            except Exception as e:
                logger.error(f"Failed to write JSON output: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ASR Streaming Client, you should start the Server \
            with pcm-input: whisperlivekit-server  --pcm-input"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./assets/jfk.flac",
        help="Audio file path or 'mic' for microphone",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="ws://localhost:8000/asr",
        help="Websocket server URI",
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=1.0,
        help="Chunk size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--simu_realtime", action="store_true", help="Simulate real-time file streaming"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save last active_transcription as JSON",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print each websocket message as raw JSON",
    )
    args = parser.parse_args()

    asyncio.run(
        run_client(
            source=args.source,
            uri=args.uri,
            chunk_size=args.chunk_size,
            simu_realtime=args.simu_realtime,
            output_json=args.output_json,
            print_raw_json=args.print_json,
        )
    )


if __name__ == "__main__":
    main()
