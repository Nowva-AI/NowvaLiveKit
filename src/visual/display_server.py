"""WebSocket hub serving the persistent Nowva display page.

Browsers subscribe on /ws and receive JSON events plus binary JPEG pose
frames. The voice agent and pose process publish on /ingest — text
messages are events broadcast to every browser, binary messages are the
latest pose frame (slow browsers skip frames, never queue them).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import websockets
from websockets.http11 import Request, Response

logger = logging.getLogger(__name__)

DISPLAY_PORT = 8768

_PAGE_HTML = Path(__file__).with_name("display.html")
_LOGO_PNG = Path(__file__).with_name("logo-white.png")

# Event types whose latest value is replayed to newly connected browsers
# so a page refresh lands in the correct state.
_STICKY_EVENT_TYPES = (
    "agent_state", "wake_word", "demo", "boot",
    "workout", "rep", "set_summary", "set_scores", "rest",
    "menu", "setup", "affect",
)

# Stickies scoped to one workout — dropped when a new "workout" event arrives
# so a refresh during a later workout doesn't replay the previous one's reps.
_WORKOUT_SCOPED_TYPES = ("rep", "set_summary", "set_scores", "rest", "menu", "setup")

# Mutually exclusive screens: showing one drops the other's sticky so a page
# refresh never replays both.
_STICKY_EXCLUSIONS = {"menu": ("setup",), "setup": ("menu",)}


class DisplayServer:
    """Runs the display hub on a daemon thread with its own event loop."""

    def __init__(self, port: int = DISPLAY_PORT, on_subscriber: Any = None) -> None:
        self._port = port
        self._on_subscriber = on_subscriber
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._subscribers: set[Any] = set()
        self._frame_waiters: dict[Any, asyncio.Event] = {}
        self._latest_frame: bytes | None = None
        self._sticky_events: dict[str, str] = {}
        self._started_event = threading.Event()

    def start(self) -> bool:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="display-server", daemon=True,
        )
        self._thread.start()
        return self._started_event.wait(timeout=5.0)

    def stop(self) -> None:
        if self._loop is not None and self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)

    def publish(self, event: dict) -> None:
        """Broadcast an event to browsers from any thread (e.g. main.py)."""
        if self._loop is None:
            return
        payload = json.dumps(event)

        def _do() -> None:
            self._remember_sticky(payload)
            self._broadcast_text(payload)

        try:
            self._loop.call_soon_threadsafe(_do)
        except RuntimeError:
            pass  # loop already closed during shutdown

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except OSError as e:
            logger.error(f"[DISPLAY] Server failed to start on :{self._port}: {e}")
            self._started_event.set()

    async def _serve(self) -> None:
        self._server = await websockets.serve(
            self._ws_handler,
            "0.0.0.0",
            self._port,
            process_request=self._http_handler,
            max_size=8 * 1024 * 1024,
        )
        self._started_event.set()
        logger.info(f"[DISPLAY] Serving on http://localhost:{self._port}")
        await self._server.wait_closed()

    async def _http_handler(self, connection: Any, request: Request) -> Response | None:
        if request.path in ("/", "/index.html"):
            return Response(
                200, "OK",
                websockets.Headers({"Content-Type": "text/html; charset=utf-8"}),
                _PAGE_HTML.read_bytes(),
            )
        if request.path == "/logo.png":
            return Response(
                200, "OK",
                websockets.Headers({"Content-Type": "image/png"}),
                _LOGO_PNG.read_bytes(),
            )
        if request.path in ("/ws", "/ingest"):
            return None
        return Response(404, "Not Found", websockets.Headers(), b"not found")

    async def _ws_handler(self, ws: Any) -> None:
        if ws.request.path == "/ingest":
            await self._ingest_loop(ws)
        else:
            await self._subscriber_loop(ws)

    async def _ingest_loop(self, ws: Any) -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self._latest_frame = message
                    for waiter in self._frame_waiters.values():
                        waiter.set()
                else:
                    self._remember_sticky(message)
                    self._broadcast_text(message)
        except websockets.ConnectionClosed:
            pass

    async def _subscriber_loop(self, ws: Any) -> None:
        if self._on_subscriber is not None:
            try:
                self._on_subscriber()
            except Exception as e:
                logger.error(f"[DISPLAY] on_subscriber callback failed: {e}")
        self._subscribers.add(ws)
        waiter = asyncio.Event()
        self._frame_waiters[ws] = waiter
        if self._latest_frame is not None:
            waiter.set()
        pump = asyncio.ensure_future(self._frame_pump(ws, waiter))
        try:
            for payload in self._sticky_events.values():
                await ws.send(payload)
            async for _ in ws:
                pass  # browsers don't send anything meaningful
        except websockets.ConnectionClosed:
            pass
        finally:
            self._subscribers.discard(ws)
            self._frame_waiters.pop(ws, None)
            pump.cancel()

    async def _frame_pump(self, ws: Any, waiter: asyncio.Event) -> None:
        try:
            while True:
                await waiter.wait()
                waiter.clear()
                frame = self._latest_frame
                if frame is not None:
                    await ws.send(frame)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    def _remember_sticky(self, payload: str) -> None:
        try:
            event_type = json.loads(payload).get("type")
        except (json.JSONDecodeError, AttributeError):
            return
        if event_type == "workout":
            for scoped in _WORKOUT_SCOPED_TYPES:
                self._sticky_events.pop(scoped, None)
        for excluded in _STICKY_EXCLUSIONS.get(event_type, ()):
            self._sticky_events.pop(excluded, None)
        if event_type in _STICKY_EVENT_TYPES:
            self._sticky_events[event_type] = payload

    def _broadcast_text(self, payload: str) -> None:
        for ws in list(self._subscribers):
            asyncio.ensure_future(self._safe_send(ws, payload))

    async def _safe_send(self, ws: Any, payload: str) -> None:
        try:
            await ws.send(payload)
        except websockets.ConnectionClosed:
            pass
