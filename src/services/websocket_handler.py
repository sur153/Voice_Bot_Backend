# ---------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See LICENSE in the project root for license information.
# --------------------------------------------------------------------------------------------

"""WebSocket handling for voice proxy connections."""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

import simple_websocket.ws  # pyright: ignore[reportMissingTypeStubs]
import websockets
import websockets.asyncio.client
from azure.identity.aio import DefaultAzureCredential,ManagedIdentityCredential
from src.config import config
from src.services.managers import AgentManager

logger = logging.getLogger(__name__)

# WebSocket constants
AZURE_VOICE_API_VERSION = "2025-10-01"
AZURE_COGNITIVE_SERVICES_DOMAIN = "cognitiveservices.azure.com"
VOICE_AGENT_ENDPOINT = "voice-agent/realtime"

# Session configuration constants
DEFAULT_MODALITIES = ["text", "audio"]
DEFAULT_TURN_DETECTION_TYPE = "azure_semantic_vad"
DEFAULT_NOISE_REDUCTION_TYPE = "azure_deep_noise_suppression"
DEFAULT_ECHO_CANCELLATION_TYPE = "server_echo_cancellation"
DEFAULT_AVATAR_CHARACTER = "lisa"
DEFAULT_AVATAR_STYLE = "casual-sitting"
DEFAULT_VOICE_NAME = "en-IN-AartiNeural" # "en-US-Ava:DragonHDLatestNeural"
DEFAULT_VOICE_TYPE = "azure-standard"

# Message types
SESSION_UPDATE_TYPE = "session.update"
PROXY_CONNECTED_TYPE = "proxy.connected"
ERROR_TYPE = "error"

# Log message truncation length
LOG_MESSAGE_MAX_LENGTH = 100


class VoiceProxyHandler:
    """Handles WebSocket proxy connections between client and Azure Voice API."""

    def __init__(self, agent_manager: AgentManager):
        """
        Initialize the voice proxy handler.

        Args:
            agent_manager: Agent manager instance
        """
        self.agent_manager = agent_manager

    async def handle_connection(self, client_ws: simple_websocket.ws.Server) -> None:
        """
        Handle a WebSocket connection from a client.

        Args:
            client_ws: The client WebSocket connection
        """

        azure_ws = None


        try:
            # current_agent_id = await self._get_agent_id_from_client(client_ws)

            azure_ws = await self._connect_to_azure()
            if not azure_ws:
                await self._send_error(client_ws, "Failed to connect to Azure Voice API")
                return

            await self._send_message(
                client_ws,
                {"type": "proxy.connected", "message": "Connected to Azure Voice API"},
            )

            await self._handle_message_forwarding(client_ws, azure_ws)

        except Exception as e:
            logger.error("Proxy error: %s", e)
            await self._send_error(client_ws, str(e))

        finally:
            if azure_ws:
                await azure_ws.close()

    async def _get_agent_id_from_client(self, client_ws: simple_websocket.ws.Server) -> Optional[str]:
        """Get agent ID from initial client message."""

        try:
            first_message: str | None = await asyncio.get_event_loop().run_in_executor(
                None,
                client_ws.receive,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )
            if first_message:
                msg = json.loads(first_message)
                if msg.get("type") == "session.update":
                    return msg.get("session", {}).get("agent_id")
        except Exception as e:
            logger.error("Error getting agent ID: %s", e)
            return None

    async def _connect_to_azure(self) -> Optional[websockets.asyncio.client.ClientConnection]:
        """Connect to Azure Voice API with appropriate configuration."""
        try:
            # agent_config = self.agent_manager.get_agent(agent_id) if agent_id else None

            azure_url = await self._build_azure_url()

            api_key = config.get("azure_openai_api_key")
            if not api_key:
                logger.error("No API key found in configuration (azure_openai_api_key)")
                return None
            client_id =  config.get("client_id")
            if client_id:
            # Use async context manager to auto-close the credential
                async with ManagedIdentityCredential(client_id=client_id) as credential:
                    token = await credential.get_token(
                        "https://ai.azure.com/.default"
                    )
                    print(token.token)
                    logger.info("Obtained agent access token")
                    headers = {"Authorization": f"Bearer {token.token}"}
            else:
                agent_access_token = (await DefaultAzureCredential().get_token("https://ai.azure.com/.default")).token
                logger.info("Obtained agent access token")
                headers = {"Authorization": f"Bearer {agent_access_token}"}
                logger.info("Connecting to Azure Voice API at URL: %s", azure_url)
            azure_ws = await websockets.connect(azure_url, additional_headers=headers)
            logger.info("Connected to Azure Voice API with agent")

            await self._send_initial_config(azure_ws)

            return azure_ws

        except Exception as e:
            logger.error("Failed to connect to Azure: %s", e)
            return None

    async def _build_azure_url(self) -> str:
        """Build the Azure WebSocket URL."""
        base_url = self._build_base_azure_url()
        return f"{base_url}&agent-name={config['agent_name']}&agent-project-name={config['azure_ai_project_name']}"



    def _build_base_azure_url(self) -> str:
        """Build the base Azure WebSocket URL."""
        resource_name = config["azure_ai_resource_name"]

        client_request_id = uuid.uuid4()

        return (
            f"wss://{resource_name}.{AZURE_COGNITIVE_SERVICES_DOMAIN}/"
            f"{VOICE_AGENT_ENDPOINT}?api-version={AZURE_VOICE_API_VERSION}"
            f"&x-ms-client-request-id={client_request_id}"
        )

    def _build_agent_specific_url(self, base_url: str, agent_id: Optional[str], agent_config: Dict[str, Any]) -> str:
        """Build URL for specific agent configuration."""
        project_name = config["azure_ai_project_name"]
        if agent_config.get("is_azure_agent"):
            return f"{base_url}&agent-id={agent_id}" f"&agent-project-name={project_name}"
        model_name = agent_config.get("model", config["model_deployment_name"])
        return f"{base_url}&model={model_name}"

    async def _send_initial_config(
        self,
        azure_ws: websockets.asyncio.client.ClientConnection
    ) -> None:
        """Send initial configuration to Azure."""
        config_message = self._build_session_config()
        print("Initial Config Message: ", config_message)
        await azure_ws.send(json.dumps(config_message))
        logger.info("Sent initial session configuration to Azure Voice API")

    def _build_session_config(self) -> Dict[str, Any]:
        """Build the base session configuration."""
        # return {
        #     "type": SESSION_UPDATE_TYPE,
        #     "session": {
        #         "modalities": DEFAULT_MODALITIES,
        #         "turn_detection": {"type": DEFAULT_TURN_DETECTION_TYPE},
        #         "input_audio_noise_reduction": {"type": DEFAULT_NOISE_REDUCTION_TYPE},
        #         "input_audio_echo_cancellation": {"type": DEFAULT_ECHO_CANCELLATION_TYPE},
        #         "avatar": {
        #             "character": DEFAULT_AVATAR_CHARACTER,
        #             "style": DEFAULT_AVATAR_STYLE,
        #         },
        #         "voice": {
        #             "name": config["azure_voice_name"],
        #             "type": config["azure_voice_type"],
        #         },
        #     },
        # }
        return {
        "type": "session.update",
        "session": {
              "modalities": DEFAULT_MODALITIES,
            # "instructions": "You are an AI assistant specializing in answering insurance-related queries. Provide accurate and helpful responses based on the provided knowledge base. If the user specifies a policy type or policy ID, retrieve the relevant details from the knowledge base.",
               "turn_detection": {
                "type": "azure_semantic_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 200,
                "remove_filler_words": True,
                "end_of_utterance_detection": {
                    "model": "semantic_detection_v1",
                    "threshold": 0.3,
                    "timeout": 1.2,
                },
            },
            "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
            "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
            "voice": {
                "name": "en-IN-AartiNeural",
                "type": "azure-standard",
                "temperature": 0,
            }
        },
    }

    def _add_local_agent_config(self, config_message: Dict[str, Any], agent_config: Dict[str, Any]) -> None:
        """Add local agent configuration to session config."""
        session = config_message["session"]
        session["model"] = agent_config.get("model", config["model_deployment_name"])
        session["instructions"] = agent_config["instructions"]
        session["temperature"] = agent_config["temperature"]
        session["max_response_output_tokens"] = agent_config["max_tokens"]

    async def _handle_message_forwarding(
        self,
        client_ws: simple_websocket.ws.Server,
        azure_ws: websockets.asyncio.client.ClientConnection,
    ) -> None:
        """Handle bidirectional message forwarding."""
        tasks = [
            asyncio.create_task(self._forward_client_to_azure(client_ws, azure_ws)),
            asyncio.create_task(self._forward_azure_to_client(azure_ws, client_ws)),
        ]

        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()

    async def _forward_client_to_azure(
        self,
        client_ws: simple_websocket.ws.Server,
        azure_ws: websockets.asyncio.client.ClientConnection,
    ) -> None:
        """Forward messages from client to Azure."""
        try:
            while True:
                message: Optional[Any] = await asyncio.get_event_loop().run_in_executor(
                    None,
                    client_ws.receive,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                )
                if message is None:
                    break
                logger.debug("Client->Azure: %s", message[:LOG_MESSAGE_MAX_LENGTH])
                await azure_ws.send(message)
        except Exception:
            logger.debug("Client connection closed during forwarding")

    async def _forward_azure_to_client(
        self,
        azure_ws: websockets.asyncio.client.ClientConnection,
        client_ws: simple_websocket.ws.Server,
    ) -> None:
        """Forward messages from Azure to client."""
        try:
            async for message in azure_ws:
                logger.debug("Azure->Client: %s", message[:LOG_MESSAGE_MAX_LENGTH])
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    client_ws.send,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                    message,
                )
        except Exception:
            logger.debug("Client connection closed during forwarding")

    async def _send_message(self, ws: simple_websocket.ws.Server, message: Dict[str, str | Dict[str, str]]) -> None:
        """Send a JSON message to a WebSocket."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                ws.send,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                json.dumps(message),
            )
        except Exception:
            pass

    async def _send_error(self, ws: simple_websocket.ws.Server, error_message: str) -> None:
        """Send an error message to a WebSocket."""
        await self._send_message(ws, {"type": "error", "error": {"message": error_message}})
