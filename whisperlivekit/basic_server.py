import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from whisperlivekit import (
    AudioProcessor,
    TranscriptionEngine,
    get_inline_ui_html,
    parse_args,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


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
                emb = diar.get_speaker_embeddings()
                payload = {"type": "speaker_embeddings", "embeddings": emb}
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

                model_meta = audio_processor.get_model_metadata()
                if model_meta:
                    payload["models"] = model_meta

                await websocket.send_json(payload)
            except Exception as e:
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
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
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
