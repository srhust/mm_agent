"""Minimal local Florence-2 grounding service.

Request:
{
  "raw_image": "/abs/path/to/image.jpg",
  "grounding_requests": [
    {"role": "Vehicle", "label": "helicopter", "grounding_query": "helicopter"}
  ],
  "task": "<CAPTION_TO_PHRASE_GROUNDING>"
}

Response:
{
  "results": [
    {
      "role": "Vehicle",
      "label": "helicopter",
      "grounding_query": "helicopter",
      "bbox": [x1, y1, x2, y2],
      "score": null,
      "grounding_status": "grounded"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mm_event_agent.grounding.florence2_hf import Florence2HFGrounder, _failed_grounding_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Florence-2 grounding HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--task", default="<CAPTION_TO_PHRASE_GROUNDING>")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _build_handler(grounder: Florence2HFGrounder) -> type[BaseHTTPRequestHandler]:
    class FlorenceHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._write_json(400, {"results": []})
                return

            raw_image = payload.get("raw_image")
            grounding_requests = payload.get("grounding_requests")
            if not isinstance(grounding_requests, list):
                self._write_json(400, {"results": []})
                return

            task = str(payload.get("task") or grounder.task_prompt or "").strip()
            if task and task != grounder.task_prompt:
                grounder.task_prompt = task

            try:
                results = grounder.execute(raw_image, grounding_requests)
            except Exception:
                results = [
                    _failed_grounding_result(request)
                    for request in grounding_requests
                    if isinstance(request, dict)
                ]
            self._write_json(200, {"results": results})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return FlorenceHandler


def main() -> None:
    args = _parse_args()
    grounder = Florence2HFGrounder(
        model_id=args.model_id,
        task_prompt=args.task,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), _build_handler(grounder))
    print(f"Florence grounding server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
