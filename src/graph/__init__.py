"""AgentCore Platform v1.0"""

# AgentRegistry discoverability: the manifest (config/agent.yaml) declares
#   module: "src.graph"
#   class:  "FNOLProcessingAgent"
# AgentRegistry resolves the entry point via importlib.import_module("src.graph")
# then getattr(module, "FNOLProcessingAgent"). Re-export the class here so the
# package (not just src.graph.graph) exposes it — otherwise auto-discovery fails
# with AttributeError even though the class exists.

from src.graph.graph import FNOLProcessingAgent

__all__ = ["FNOLProcessingAgent"]
