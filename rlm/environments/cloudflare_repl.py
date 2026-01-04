"""
Cloudflare Sandbox REPL environment that runs Python code in Cloudflare Sandboxes.

Requires a deployed Cloudflare Worker implementing the RLM sandbox API.
See the plan file for the Worker API contract.
"""

import json
import threading
import time
import uuid

import requests

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import IsolatedEnv


class CloudflareREPL(IsolatedEnv):
    """
    Cloudflare Sandbox environment - runs Python code in Cloudflare Sandboxes.

    Requires a deployed Cloudflare Worker implementing the RLM sandbox API.
    Uses HTTP polling pattern (like ModalREPL) for LLM callbacks.

    The Worker must implement these endpoints:
    - POST /session - Create/get a sandbox session
    - POST /execute - Execute Python code
    - POST /context - Load context data
    - GET /pending - Get pending LLM requests
    - POST /respond - Submit LLM response
    - DELETE /session - Cleanup sandbox
    """

    def __init__(
        self,
        worker_url: str,
        session_id: str | None = None,
        timeout: int = 300,
        poll_interval: float = 0.1,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        **kwargs,
    ):
        """
        Initialize CloudflareREPL.

        Args:
            worker_url: URL of the deployed Cloudflare Worker (e.g., https://rlm-sandbox.example.workers.dev)
            session_id: Optional session ID to reuse. If not provided, a new session is created.
            timeout: Request timeout in seconds
            poll_interval: How often to poll for pending LLM requests (seconds)
            lm_handler_address: Address of the LM handler for routing LLM requests
            context_payload: Initial context to load into the sandbox
            setup_code: Code to execute during setup
        """
        super().__init__(**kwargs)

        self.worker_url = worker_url.rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.lm_handler_address = lm_handler_address

        self.poller_thread: threading.Thread | None = None
        self.poller_stop = threading.Event()
        self.pending_llm_calls: list[RLMChatCompletion] = []
        self._calls_lock = threading.Lock()

        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)

        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Create or retrieve a sandbox session."""
        try:
            response = requests.post(
                f"{self.worker_url}/session",
                json={"session_id": self.session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data.get("session_id", self.session_id)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create Cloudflare sandbox session: {e}")

        # Start polling thread if we have an LM handler
        if self.lm_handler_address:
            self.poller_stop.clear()
            self.poller_thread = threading.Thread(target=self._poll_pending, daemon=True)
            self.poller_thread.start()

    def _poll_pending(self):
        """Poll the Worker for pending LLM requests and handle them."""
        while not self.poller_stop.is_set():
            try:
                resp = requests.get(
                    f"{self.worker_url}/pending",
                    params={"session_id": self.session_id},
                    timeout=5,
                )
                pending = resp.json().get("pending", [])

                for item in pending:
                    request_id = item["id"]
                    req_data = item["request"]

                    # Handle the request
                    response = self._handle_llm_request(req_data)

                    # Send response back
                    requests.post(
                        f"{self.worker_url}/respond",
                        json={
                            "session_id": self.session_id,
                            "id": request_id,
                            "response": response,
                        },
                        timeout=10,
                    )

            except requests.exceptions.RequestException:
                pass
            except Exception:
                pass

            time.sleep(self.poll_interval)

    def _handle_llm_request(self, req_data: dict) -> dict:
        """Handle an LLM request from the sandbox."""
        req_type = req_data.get("type")
        model = req_data.get("model")

        if req_type == "single":
            prompt = req_data.get("prompt")
            request = LMRequest(prompt=prompt, model=model)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return {"error": response.error}

            # Track the call
            with self._calls_lock:
                self.pending_llm_calls.append(response.chat_completion)

            return {"response": response.chat_completion.response}

        elif req_type == "batched":
            prompts = req_data.get("prompts", [])
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model
            )

            results = []
            for resp in responses:
                if not resp.success:
                    results.append(f"Error: {resp.error}")
                else:
                    with self._calls_lock:
                        self.pending_llm_calls.append(resp.chat_completion)
                    results.append(resp.chat_completion.response)

            return {"responses": results}

        return {"error": "Unknown request type"}

    def load_context(self, context_payload: dict | list | str):
        """Load context into the sandbox environment.

        Creates a 'context' variable in the sandbox that contains the payload.
        Matches the ModalREPL pattern.
        """
        import json as json_module

        if isinstance(context_payload, str):
            # Escape the string for embedding in Python code
            escaped = context_payload.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            context_code = f'context = """{escaped}"""'
        else:
            # Convert to JSON and parse in the sandbox
            context_json = json_module.dumps(context_payload)
            escaped_json = context_json.replace("\\", "\\\\").replace("'", "\\'")
            context_code = f"import json; context = json.loads('{escaped_json}')"

        self.execute_code(context_code)

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the Cloudflare sandbox and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls
        with self._calls_lock:
            self.pending_llm_calls.clear()

        try:
            response = requests.post(
                f"{self.worker_url}/execute",
                json={
                    "session_id": self.session_id,
                    "code": code,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            return REPLResult(
                stdout="",
                stderr=f"Execution failed: {e}",
                locals={},
                execution_time=time.perf_counter() - start_time,
                rlm_calls=[],
            )

        # Collect LLM calls made during this execution
        with self._calls_lock:
            pending_calls = self.pending_llm_calls.copy()
            self.pending_llm_calls.clear()

        execution_time = time.perf_counter() - start_time

        return REPLResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            locals=result.get("locals", {}),
            execution_time=result.get("execution_time", execution_time),
            rlm_calls=pending_calls,
        )

    def cleanup(self):
        """Terminate the sandbox session and stop polling."""
        # Stop the poller thread
        if self.poller_thread is not None:
            self.poller_stop.set()
            self.poller_thread.join(timeout=2)
            self.poller_thread = None

        # Delete the session
        try:
            requests.delete(
                f"{self.worker_url}/session",
                json={"session_id": self.session_id},
                timeout=10,
            )
        except requests.RequestException:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
