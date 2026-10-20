from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict

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
        obj = torch.load(buffer, map_location='cpu', weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """An inference server that spin up a ZeroMQ socket and listen for incoming
    requests.

    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = '*', port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://{host}:{port}')
        self._endpoints: Dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint('ping', self._handle_ping, requires_input=False)
        self.register_endpoint('kill', self._kill_server, requires_input=False)

    def _kill_server(self):
        """Kill the server."""
        self.running = False

    def _handle_ping(self) -> Dict[str, Any]:
        """Simple ping handler that returns a success message."""
        return {'status': 'ok', 'message': 'Server is running'}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        """Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get('endpoint', 'inference')

                if endpoint not in self._endpoints:
                    raise ValueError(f'Unknown endpoint: {endpoint}')

                handler = self._endpoints[endpoint]
                result = handler.handler(request.get('data', {})) if handler.requires_input else handler.handler()
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                import traceback

                print(traceback.format_exc())
                self.socket.send(b'ERROR')


class RobotInferenceServer(BaseInferenceServer):
    """Server with three endpoints for real robot policies."""

    def __init__(self, model, host: str = '*', port: int = 5555):
        super().__init__(host, port)
        self.register_endpoint('inference', model.inference)

    @staticmethod
    def start_server(policy, port: int) -> None:
        server = RobotInferenceServer(policy, port=port)
        server.run()
