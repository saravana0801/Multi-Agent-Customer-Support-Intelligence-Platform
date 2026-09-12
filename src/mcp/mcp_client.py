"""
Model Context Protocol (MCP) Client Adapter.
Connects the Multi-Agent CrewAI Flow, agents, and API to the CustomerSupportMCPServer.
Dispatches tool invocations and retrieves MCP resources using the official MCP schema.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Callable

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.mcp.support_mcp_server import (
    lookup_order,
    search_faq_policies,
    search_golden_resolutions,
    verify_refund_eligibility,
    get_return_policy_resource,
    get_metrics_resource,
    mcp as server_instance,
)


class MCPClientAdapter:
    """
    Client adapter facilitating standard MCP (Model Context Protocol) tool
    discovery, parameter validation, and execution for CrewAI agents and Flows.
    """

    def __init__(self):
        self.server_name = server_instance.name
        self._tools_registry: Dict[str, Callable] = {
            "lookup_order": lookup_order,
            "search_faq_policies": search_faq_policies,
            "search_golden_resolutions": search_golden_resolutions,
            "verify_refund_eligibility": verify_refund_eligibility,
        }
        self._resources_registry: Dict[str, Callable] = {
            "support://policies/return-and-refund": get_return_policy_resource,
            "support://metrics/summary": get_metrics_resource,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns all MCP tools exposed by the server with JSON schemas."""
        tools_metadata = [
            {
                "name": "lookup_order",
                "description": "Looks up order carrier, fulfillment status, and items from the OMS.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
            {
                "name": "search_faq_policies",
                "description": "Performs semantic vector search over official store FAQ policies in ChromaDB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {"type": "string"},
                        "limit": {"type": "integer", "default": 2},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_golden_resolutions",
                "description": "Performs semantic vector search over proven past customer ticket resolutions (CSAT >= 4).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {"type": "string"},
                        "limit": {"type": "integer", "default": 2},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "verify_refund_eligibility",
                "description": "Validates financial refund requests and checks if supervisor HITL approval is required (> $50).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["order_id", "amount"],
                },
            },
        ]
        return tools_metadata

    def list_resources(self) -> List[Dict[str, str]]:
        """Returns all MCP resources exposed by the server."""
        return [
            {
                "uri": "support://policies/return-and-refund",
                "name": "Store Return & Refund Policy",
                "mimeType": "text/markdown",
            },
            {
                "uri": "support://metrics/summary",
                "name": "System SLA and Performance Metrics",
                "mimeType": "application/json",
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes an MCP tool call through the adapter.
        """
        if name not in self._tools_registry:
            raise ValueError(f"Unknown MCP tool: '{name}'. Available: {list(self._tools_registry.keys())}")

        handler = self._tools_registry[name]
        try:
            result = handler(**arguments)
            return {
                "status": "success",
                "tool": name,
                "server": self.server_name,
                "result": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "tool": name,
                "server": self.server_name,
                "error": str(e),
            }

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Reads an MCP resource from the server.
        """
        if uri not in self._resources_registry:
            raise ValueError(f"Unknown MCP resource URI: '{uri}'. Available: {list(self._resources_registry.keys())}")

        handler = self._resources_registry[uri]
        content = handler()
        return {
            "uri": uri,
            "server": self.server_name,
            "content": content,
        }


# Singleton client instance
mcp_client = MCPClientAdapter()


if __name__ == "__main__":
    print(f"=== MCP CLIENT CONNECTED TO: {mcp_client.server_name} ===")
    print("\n1. Discovered Tools:")
    for t in mcp_client.list_tools():
        print(f"  • {t['name']}: {t['description']}")

    print("\n2. Discovered Resources:")
    for r in mcp_client.list_resources():
        print(f"  • {r['name']} ({r['uri']})")

    print("\n3. Testing Tool Call: lookup_order('ORD1176788')...")
    res = mcp_client.call_tool("lookup_order", {"order_id": "ORD1176788"})
    print("  -> Result:", res)

    print("\n4. Testing Tool Call: verify_refund_eligibility('ORD1176788', 75.0)...")
    res2 = mcp_client.call_tool("verify_refund_eligibility", {"order_id": "ORD1176788", "amount": 75.0})
    print("  -> Result:", res2)
