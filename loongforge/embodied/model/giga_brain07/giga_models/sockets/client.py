from io import BytesIO
from typing import Any, Dict, Optional

import torch
import zmq


class TorchSerializer:
    """Serialize/deserialize Python objects using torch.save/torch.load."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes):
        buffer = BytesIO(data)
        # The protocol supports arbitrary Python objects and therefore requires a trusted peer.
        obj = torch.load(buffer, map_location='cpu', weights_only=False)
        return obj


class BaseInferenceClient:
    """ZeroMQ request client with simple endpoint calling utility."""

    def __init__(self, host: str = 'localhost', port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self) -> None:
        """Kill the server."""
        self.call_endpoint('kill', requires_input=False)

    def call_endpoint(self, endpoint: str, data: Optional[dict] = None, requires_input: bool = True) -> dict:
        """Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b'ERROR':
            raise RuntimeError('Server error')
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction."""
        self.socket.close()
        self.context.term()


class RobotInferenceClient(BaseInferenceClient):
    """Client for communicating with the RobotInferenceServer."""

    def inference(self, observations: Dict[str, Any]):
        return self.call_endpoint('inference', observations)
