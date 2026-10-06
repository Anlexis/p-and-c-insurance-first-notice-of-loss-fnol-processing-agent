"""AgentCore Platform v1.0"""

# INS-C2-015 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the FNOL coverage RAG pipeline:
#
#   START
#     → input_validate   (InputValidateNode)
#     → retrieve         (RetrieveNode)
#     → rerank_filter    (RerankFilterNode)
#     → generate_answer  (GenerateAnswerNode)
#     → output_format    (OutputFormatNode)
#     → END
#
# Called by FNOLRetrievalGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
#   ✅ Implements all 7 BaseGraph abstract methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Does NOT register initialize / finalize (outer backbone concerns)
#   ✅ All inner nodes declare required_trust_level = TrustLevel.ANONYMOUS
#   ✅ get_output() designed together with FNOLRetrievalGraphNode.merge_output()
#   ✅ All inner node constructors are empty-parens (no constructor arguments):
#      the node contract is execute(self, state) -> dict, so declared settings
#      travel through state via _extra_initial_state() below
#   ❌ No platform-SDK imports (framework/ and shared/ only)

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-015.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by FNOLRetrievalGraphNode.get_subgraph() in graph.py.

    Pipeline (linear RAG):
        START
          → input_validate   (InputValidateNode)
          → retrieve         (RetrieveNode)
          → rerank_filter    (RerankFilterNode)
          → generate_answer  (GenerateAnswerNode)
          → output_format    (OutputFormatNode)
          → END

    All nodes are FunctionNode subclasses with ANONYMOUS trust_level.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ins_c2_015_fnol_coverage_rag_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory settings: every declared value is optional and bounded.

        FNOLRetrievalGraphNode._parent_config() already dropped any entry that
        was absent, mistyped, non-finite or out of range, so whatever arrives
        here is either a usable value or missing — and a missing value means
        the consuming node keeps its own documented default.
        """
        return None

    # ── Initial state ─────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner graph's initial state.

        Two things cross the outer→inner boundary here, and neither can travel
        any other way:

        - ``input_context`` — the framework's GraphNode.execute() invokes this
          subgraph without forwarding it, so it is stashed by the outer node's
          extract_input() and picked up here (src/graph/context_bridge.py).
        - ``runtime_settings`` — the declared, already-validated retrieval
          settings. Node constructors take no arguments and ``execute(state)``
          takes no config argument, so state is the route by which a value in
          config/config.yaml reaches a domain node.
        """
        return {
            "input_context": get_caller_input_context(),
            "runtime_settings": to_json(dict(self.config)),
        }

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear FNOL coverage RAG topology.

        Linear flow:
            input_validate → retrieve → rerank_filter → generate_answer
            → output_format → END.

        No conditional branching — every path through the RAG pipeline is
        linear, so route() satisfies the abstract contract but is not wired
        into any edge.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph abstract contract.

        Linear topology; add_conditional_edges() is not used, so this method is
        never called at runtime. It is annotated with this graph's own State so
        that, were it ever wired into a conditional edge, the routing fields it
        reads would not be projected away. Returns END on error so an unexpected
        invocation does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by FNOLRetrievalGraphNode.merge_output()
        in graph.py as the `sub_result` argument.  Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output() emits:   "fnol_response", "grounded_answer",
                                        "reranked_docs", "status"
            Outer merge_output() reads: sub_result.get(...) for each key above.
        """
        return {
            "fnol_response": state.get("fnol_response"),
            "grounded_answer": state.get("grounded_answer"),
            "reranked_docs": state.get("reranked_docs"),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
