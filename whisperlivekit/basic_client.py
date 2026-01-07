import asyncio
import websockets
import numpy as np
import librosa
import sounddevice as sd
import json
import time
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


async def send_audio(
    websocket, source, chunk_size_s=1.0, sample_rate=16000, simu_realtime=False
):
    """Send audio chunks (file or mic) to websocket"""
    if source == "mic":
        logger.info("Streaming from microphone...")
        q = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            q.put_nowait(indata.copy())

        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", callback=callback
        ):
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                await _send_chunk(websocket, chunk, sample_rate)
                await asyncio.sleep(chunk_size_s)

    else:
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
            await _send_chunk(websocket, chunk, sr)
            if simu_realtime:
                await asyncio.sleep(chunk_size_s)

        await websocket.send(b"")  # End of stream
        return duration


async def _send_chunk(websocket, chunk, sr_in):
    """Encode PCM chunk to bytes and send"""
    chunk_int16 = (chunk * 32768).astype(np.int16)
    await websocket.send(chunk_int16.tobytes())


async def receive_updates(
    websocket,
    first_token_event,
    collected_lines=None,
    collected_embeddings: dict | None = None,
    collected_speaker_ids: dict | None = None,
    collected_models: dict | None = None,
    print_raw_json: bool = False,
):
    """Receive server responses, mark first token, and collect lines/embeddings/ids"""
    while True:
        try:
            msg = await websocket.recv()
            resp = json.loads(msg)
            if print_raw_json:
                print(json.dumps(resp, ensure_ascii=False))
            if "lines" in resp and len(resp["lines"]) > 0:
                for line in resp["lines"]:
                    sid = ""
                    if "speaker_id" in line:
                        sid = f" [{line['speaker_id']}]"
                    prefix = "\r" if not line.get("final") else "\n"
                    print(
                        f"{prefix}{line['start']} - {line['end']} Speaker {line['speaker']}{sid}: {line['text']}",
                        end="" if not line.get("final") else "\n",
                        flush=True,
                    )
                    if collected_lines is not None:
                        collected_lines.append(line)
                    if not first_token_event.is_set():
                        first_token_event.set()
            for k, v in resp.items():
                if k != "lines":
                    if k == "type":
                        print(f"\n{k}: {v}")
                        if (
                            v == "speaker_embeddings"
                            and collected_embeddings is not None
                        ):
                            if "embeddings" in resp and isinstance(
                                resp["embeddings"], dict
                            ):
                                collected_embeddings.update(resp["embeddings"])
                            if (
                                "speaker_ids" in resp
                                and isinstance(resp["speaker_ids"], dict)
                                and collected_speaker_ids is not None
                            ):
                                collected_speaker_ids.update(resp["speaker_ids"])
                        continue
                    if k == "speaker_ids" and collected_speaker_ids is not None:
                        collected_speaker_ids.update(v)
                    if k == "models" and collected_models is not None:
                        collected_models.update(v)

            if "type" in resp and resp["type"] == "ready_to_stop":
                break

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")
            break
        except Exception as e:
            logger.error(f"Error receiving updates: {e}")
            break


async def test_server(
    source,
    host="localhost",
    port=8000,
    chunk_size=1.0,
    simu_realtime=False,
    output_json: str | None = None,
    print_raw_json: bool = False,
):
    """Main pipeline: send + receive + metrics"""
    uri = f"ws://{host}:{port}/asr"
    async with websockets.connect(uri) as ws:
        logger.info(f"Connected to {uri}")
        first_token_event = asyncio.Event()
        collected_lines: list[dict] = []
        collected_embeddings: dict = {}
        collected_speaker_ids: dict = {}
        collected_models: dict = {}
        recv_task = asyncio.create_task(
            receive_updates(
                ws,
                first_token_event,
                collected_lines,
                collected_embeddings,
                collected_speaker_ids,
                collected_models,
                print_raw_json=print_raw_json,
            )
        )

        start_time = time.time()
        duration = await send_audio(
            ws, source, chunk_size_s=chunk_size, simu_realtime=simu_realtime
        )

        try:
            await asyncio.wait_for(first_token_event.wait(), timeout=30)
            first_latency = time.time() - start_time
        except asyncio.TimeoutError:
            first_latency = None

        await recv_task
        total_time = time.time() - start_time
        rtf = total_time / duration if duration else None

        print("\n========== METRICS ==========")
        print(
            f"First Token Latency: {first_latency:.3f}s"
            if first_latency
            else "No token received"
        )
        print(f"Total Time: {total_time:.3f}s")
        print(f"Real Time Factor: {rtf:.3f}" if rtf else "RTF: undefined (mic input)")
        print("=============================\n")

        # Optionally write transcription, embeddings, and metrics to JSON file
        if output_json:
            try:
                out_path = Path(output_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "source": source,
                    "host": host,
                    "port": port,
                    "chunk_size": chunk_size,
                    "simu_realtime": simu_realtime,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "transcription": {"lines": collected_lines},
                    "speaker_embeddings": collected_embeddings or {},
                    "speaker_ids": collected_speaker_ids or {},
                    "models": collected_models or {},
                    "metrics": {
                        "first_token_latency": first_latency,
                        "total_time": total_time,
                        "real_time_factor": rtf,
                        "duration": duration,
                    },
                }
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved transcription JSON to {out_path}")
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
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
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
        help="Optional path to save transcription and metrics as JSON",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print each websocket message as raw JSON",
    )
    args = parser.parse_args()

    asyncio.run(
        test_server(
            source=args.source,
            host=args.host,
            port=args.port,
            chunk_size=args.chunk_size,
            simu_realtime=args.simu_realtime,
            output_json=args.output_json,
            print_raw_json=args.print_json,
        )
    )


if __name__ == "__main__":
    main()
