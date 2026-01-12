import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from whisperlivekit import (
    AudioProcessor,
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)
from whisperlivekit.speaker_store import SpeakerStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None
speaker_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine, speaker_store
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    try:
        speaker_store = SpeakerStore(args.chroma_path, args.chroma_collection)
    except Exception as e:
        logger.warning(f"Failed to initialize SpeakerStore: {e}")
        speaker_store = None
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeakerUpdateRequest(BaseModel):
    name: str | None = None
    source_model: str | None = None
    recording_id: str | None = None
    embedding: list[float] | None = None


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _summarize_diarization_state(diar) -> dict:
    summary = {"class": diar.__class__.__name__}
    stream_state = getattr(diar, "streaming_state", None)
    if stream_state is not None:
        for name in ("spkcache_lengths", "fifo_lengths"):
            val = getattr(stream_state, name, None)
            if val is not None:
                if hasattr(val, "detach") and hasattr(val, "cpu") and hasattr(val, "tolist"):
                    summary[name] = val.detach().cpu().tolist()
                elif hasattr(val, "tolist"):
                    summary[name] = val.tolist()
                else:
                    summary[name] = str(val)
    total_preds = getattr(diar, "total_preds", None)
    if total_preds is not None:
        if hasattr(total_preds, "shape"):
            summary["total_preds_shape"] = tuple(total_preds.shape)
        else:
            summary["total_preds_shape"] = str(getattr(total_preds, "shape", None))
    buffer_audio = getattr(diar, "buffer_audio", None)
    if buffer_audio is not None:
        if hasattr(buffer_audio, "__len__"):
            summary["buffer_audio_len"] = int(len(buffer_audio))
        else:
            summary["buffer_audio_len"] = str(buffer_audio)
    return summary


def _speaker_payload(speaker: dict, include_embedding: bool) -> dict:
    metadata = speaker.get("metadata") or {}
    payload = {
        "id": speaker["id"],
        "name": metadata.get("name"),
        "source_model": metadata.get("source_model"),
        "recording_id": metadata.get("recording_id"),
    }
    if include_embedding:
        payload["embedding"] = speaker.get("embedding")
    return payload


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/speakers/{speaker_id}")
async def get_speaker(speaker_id: str, include_embedding: bool = False):
    if speaker_store is None:
        raise HTTPException(status_code=503, detail="Speaker store unavailable")
    speaker = speaker_store.get_speaker(speaker_id, include_embedding=include_embedding)
    if speaker is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return _speaker_payload(speaker, include_embedding)


@app.patch("/speakers/{speaker_id}")
async def update_speaker(speaker_id: str, update: SpeakerUpdateRequest):
    if speaker_store is None:
        raise HTTPException(status_code=503, detail="Speaker store unavailable")
    if (
        update.name is None
        and update.source_model is None
        and update.recording_id is None
        and update.embedding is None
    ):
        raise HTTPException(status_code=422, detail="No fields provided for update")
    try:
        speaker = speaker_store.update_speaker(
            speaker_id,
            name=update.name,
            source_model=update.source_model,
            recording_id=update.recording_id,
            embedding=update.embedding,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if speaker is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return _speaker_payload(speaker, include_embedding=update.embedding is not None)


@app.delete("/speakers/{speaker_id}", status_code=204)
async def delete_speaker(speaker_id: str):
    if speaker_store is None:
        raise HTTPException(status_code=503, detail="Speaker store unavailable")
    if not speaker_store.delete_speaker(speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return None


async def handle_websocket_results(
    websocket, results_generator, audio_processor: AudioProcessor
):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Preparing final artifacts...")
        # Optionally flush remaining audio and send speaker embeddings before ready_to_stop.
        # Fail-first inside embedding functions, but ensure client receives an error event and can stop.
        if args.emit_speaker_embeddings and getattr(
            audio_processor, "diarization", None
        ):
            diar = audio_processor.diarization
            # Flush all remaining buffered audio (zero-pad last chunk as needed)
            if hasattr(diar, "flush_remaining"):
                while diar.flush_remaining():
                    pass
            try:
                if not hasattr(diar, "get_speaker_embeddings"):
                    raise AttributeError(
                        "Diarization backend does not provide get_speaker_embeddings()"
                    )
                await audio_processor.maybe_update_speakers(finalize=True)
                emb = diar.get_speaker_embeddings()
                if not emb:
                    logger.warning(
                        "No speaker embeddings returned at end-of-stream. "
                        "diarization_backend=%s diag=%s",
                        args.diarization_backend,
                        _summarize_diarization_state(diar),
                    )
                payload = {"type": "speaker_embeddings"}
                try:
                    speaker_ids = audio_processor.update_speaker_ids_from_embeddings(
                        emb
                    )
                    if speaker_ids:
                        payload["speaker_ids"] = speaker_ids
                        payload["speaker_id_bits"] = audio_processor.speaker_hash_bits
                except Exception as id_exc:
                    logger.warning(
                        f"Failed to derive speaker IDs from embeddings: {id_exc}"
                    )
                if audio_processor.return_candidates and audio_processor.speaker_candidates:
                    payload["speaker_candidates"] = audio_processor.speaker_candidates
                    payload["candidates_final"] = True

                model_meta = audio_processor.get_model_metadata()
                if model_meta:
                    payload["models"] = model_meta

                await websocket.send_json(payload)
            except Exception as e:
                logger.exception(
                    "speaker_embeddings_failed: %s (backend=%s diag=%s)",
                    e,
                    args.diarization_backend,
                    _summarize_diarization_state(diar),
                )
                await websocket.send_json(
                    {"type": "error", "error": f"speaker_embeddings_failed: {e}"}
                )
        logger.info("Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected while handling results (client likely closed connection)."
        )
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    params = websocket.query_params
    return_candidates = _parse_bool(params.get("return_candidates"), default=False)
    candidate_topk = _parse_int(params.get("topk"), default=3)
    candidate_threshold = _parse_float(
        params.get("candidates_threshold"), default=0.75
    )
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        speaker_store=speaker_store,
        return_candidates=return_candidates,
        candidate_topk=candidate_topk,
        candidate_threshold=candidate_threshold,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json(
            {
                "type": "config",
                "useAudioWorklet": bool(args.pcm_input),
                "models": audio_processor.get_model_metadata(),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator, audio_processor)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(
                f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(
            f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True
        )
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        # Signal end-of-stream to downstream processors to let embeddings flush.
        try:
            await audio_processor.process_audio(b"")
        except Exception as e:
            logger.warning(f"Failed to push final stop signal: {e}")

        if not websocket_task.done():
            try:
                await asyncio.wait_for(websocket_task, timeout=5.0)
            except asyncio.TimeoutError:
                websocket_task.cancel()
                try:
                    await websocket_task
                except asyncio.CancelledError:
                    logger.info(
                        "WebSocket results handler task was cancelled after timeout."
                    )
            except asyncio.CancelledError:
                logger.info("WebSocket results handler task was cancelled.")
            except Exception as e:
                logger.warning(
                    f"Exception while awaiting websocket_task completion: {e}"
                )

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command."""
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError(
                "Both --ssl-certfile and --ssl-keyfile must be specified together."
            )
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = {
            **uvicorn_kwargs,
            "forwarded_allow_ips": args.forwarded_allow_ips,
        }

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
