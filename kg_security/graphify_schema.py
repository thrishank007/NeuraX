"""
Graphify graph.json schema helpers.

Parses NetworkX node-link JSON produced by the Graphify CLI and exposes
serializable DTOs for the API/UI layer. Does not import Graphify internals.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger


@dataclass
class GraphifyNodeDTO:
    id: str
    label: str = ""
    file_type: str = ""
    source_file: str = ""
    source_location: str = ""
    community: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphifyEdgeDTO:
    source: str
    target: str
    relation: str = ""
    confidence: str = ""
    confidence_score: Optional[float] = None
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphifyGraph:
    nodes: List[GraphifyNodeDTO]
    edges: List[GraphifyEdgeDTO]
    directed: bool = True
    multigraph: bool = False
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "directed": self.directed,
            "multigraph": self.multigraph,
            "raw_meta": self.raw_meta,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


class GraphParseError(ValueError):
    """Raised when graph.json is invalid or unsafe."""


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_edge_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "links" in payload and isinstance(payload["links"], list):
        return payload["links"]
    if "edges" in payload and isinstance(payload["edges"], list):
        return payload["edges"]
    return []


def parse_graph_payload(payload: Any) -> GraphifyGraph:
    """Validate and parse a Graphify/NetworkX node-link JSON object."""
    if not isinstance(payload, dict):
        raise GraphParseError("Top-level graph JSON must be an object")

    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, list):
        raise GraphParseError("Graph JSON 'nodes' must be a list")

    edges_raw = _normalize_edge_list(payload)
    if "links" not in payload and "edges" not in payload:
        # Empty edge list is allowed; missing both keys is still ok for empty graphs
        edges_raw = []

    nodes: List[GraphifyNodeDTO] = []
    node_ids: set[str] = set()
    for idx, node in enumerate(nodes_raw):
        if not isinstance(node, dict):
            raise GraphParseError(f"Node at index {idx} is not an object")
        node_id = node.get("id")
        if node_id is None or str(node_id).strip() == "":
            raise GraphParseError(f"Node at index {idx} is missing a stable 'id'")
        nid = str(node_id)
        if nid in node_ids:
            # Keep first occurrence; skip duplicates deterministically
            continue
        node_ids.add(nid)
        community = node.get("community")
        nodes.append(
            GraphifyNodeDTO(
                id=nid,
                label=_as_str(node.get("label") or node.get("name") or nid),
                file_type=_as_str(node.get("file_type") or node.get("type") or node.get("kind")),
                source_file=_as_str(node.get("source_file") or node.get("file") or node.get("path")),
                source_location=_as_str(node.get("source_location") or node.get("location")),
                community=None if community is None else str(community),
                metadata={
                    k: v
                    for k, v in node.items()
                    if k
                    not in {
                        "id",
                        "label",
                        "name",
                        "file_type",
                        "type",
                        "kind",
                        "source_file",
                        "file",
                        "path",
                        "source_location",
                        "location",
                        "community",
                    }
                },
            )
        )

    edges: List[GraphifyEdgeDTO] = []
    for idx, edge in enumerate(edges_raw):
        if not isinstance(edge, dict):
            raise GraphParseError(f"Edge at index {idx} is not an object")
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            raise GraphParseError(f"Edge at index {idx} missing source/target")
        sid, tid = str(source), str(target)
        if sid not in node_ids or tid not in node_ids:
            raise GraphParseError(
                f"Edge at index {idx} references missing node id(s): {sid} -> {tid}"
            )
        conf = edge.get("confidence")
        conf_score = edge.get("confidence_score")
        if conf_score is None and isinstance(conf, (int, float)):
            conf_score = conf
            conf_label = ""
        else:
            conf_label = _as_str(conf)
        edges.append(
            GraphifyEdgeDTO(
                source=sid,
                target=tid,
                relation=_as_str(edge.get("relation") or edge.get("type") or edge.get("label")),
                confidence=conf_label,
                confidence_score=_optional_float(conf_score),
                source_file=_as_str(edge.get("source_file") or edge.get("file")),
                metadata={
                    k: v
                    for k, v in edge.items()
                    if k
                    not in {
                        "source",
                        "target",
                        "relation",
                        "type",
                        "label",
                        "confidence",
                        "confidence_score",
                        "source_file",
                        "file",
                    }
                },
            )
        )

    directed = bool(payload.get("directed", True))
    multigraph = bool(payload.get("multigraph", False))
    raw_meta = {
        k: v
        for k, v in payload.items()
        if k not in {"nodes", "links", "edges", "directed", "multigraph"}
    }
    return GraphifyGraph(
        nodes=nodes,
        edges=edges,
        directed=directed,
        multigraph=multigraph,
        raw_meta=raw_meta,
    )


def load_graph_json(
    path: Path,
    *,
    max_bytes: int = 50 * 1024 * 1024,
) -> GraphifyGraph:
    """Load and validate graph.json from disk with size safety checks."""
    if not path.exists():
        raise GraphParseError(f"Graph file not found: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise GraphParseError(
            f"Graph file exceeds size limit ({size} bytes > {max_bytes} bytes)"
        )
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphParseError(f"Malformed graph JSON: {exc}") from exc
    except OSError as exc:
        raise GraphParseError(f"Could not read graph file: {exc}") from exc
    return parse_graph_payload(payload)


def to_networkx(graph: GraphifyGraph) -> nx.Graph:
    """Build a NetworkX graph from DTOs for analytics."""
    if graph.multigraph:
        G: nx.Graph = nx.MultiDiGraph() if graph.directed else nx.MultiGraph()
    else:
        G = nx.DiGraph() if graph.directed else nx.Graph()

    for node in graph.nodes:
        G.add_node(
            node.id,
            label=node.label,
            file_type=node.file_type,
            source_file=node.source_file,
            source_location=node.source_location,
            community=node.community,
            **{k: v for k, v in node.metadata.items() if isinstance(v, (str, int, float, bool))},
        )
    for edge in graph.edges:
        G.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            confidence=edge.confidence,
            confidence_score=edge.confidence_score,
            source_file=edge.source_file,
        )
    return G


def compute_graph_stats(graph: GraphifyGraph) -> Dict[str, Any]:
    """Aggregate statistics for UI tables."""
    G = to_networkx(graph)
    type_counts = Counter(n.file_type or "unknown" for n in graph.nodes)
    relation_counts = Counter(e.relation or "unknown" for e in graph.edges)
    confidence_counts = Counter(
        (e.confidence or "unspecified") if e.confidence else "unspecified"
        for e in graph.edges
    )
    community_counts = Counter(
        (n.community if n.community is not None else "none") for n in graph.nodes
    )
    source_counts = Counter(n.source_file or "unknown" for n in graph.nodes if n.source_file)

    degree_items: List[Tuple[str, int, str]] = []
    for nid, deg in G.degree():
        label = G.nodes[nid].get("label", nid)
        degree_items.append((str(nid), int(deg), str(label)))
    degree_items.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [
        {"id": nid, "label": label, "degree": deg}
        for nid, deg, label in degree_items[:20]
    ]

    undirected = G.to_undirected() if G.is_directed() else G
    components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
    component_sizes = sorted((len(c) for c in components), reverse=True)

    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "directed": graph.directed,
        "multigraph": graph.multigraph,
        "community_count": len([c for c in community_counts if c != "none"]),
        "counts_by_node_type": dict(type_counts.most_common()),
        "counts_by_relation": dict(relation_counts.most_common()),
        "counts_by_confidence": dict(confidence_counts.most_common()),
        "counts_by_community": dict(community_counts.most_common(50)),
        "top_nodes_by_degree": top_nodes,
        "connected_components": {
            "count": len(components),
            "largest": component_sizes[0] if component_sizes else 0,
            "sizes_top10": component_sizes[:10],
        },
        "source_file_coverage": dict(source_counts.most_common(50)),
        "source_file_count": len(source_counts),
    }


def filter_graph(
    graph: GraphifyGraph,
    *,
    node_type: Optional[str] = None,
    community: Optional[str] = None,
    source_file: Optional[str] = None,
    confidence: Optional[str] = None,
    min_confidence_score: Optional[float] = None,
) -> GraphifyGraph:
    """Return a filtered copy of the graph DTOs."""
    nodes = graph.nodes
    if node_type:
        nodes = [n for n in nodes if (n.file_type or "").lower() == node_type.lower()]
    if community is not None and community != "":
        nodes = [n for n in nodes if str(n.community) == str(community)]
    if source_file:
        sf = source_file.lower()
        nodes = [n for n in nodes if sf in (n.source_file or "").lower()]

    allowed = {n.id for n in nodes}
    edges = [e for e in graph.edges if e.source in allowed and e.target in allowed]
    if confidence:
        conf = confidence.lower()
        edges = [e for e in edges if conf in (e.confidence or "").lower()]
    if min_confidence_score is not None:
        edges = [
            e
            for e in edges
            if e.confidence_score is not None and e.confidence_score >= min_confidence_score
        ]
        # Keep nodes that appear in remaining edges or still match filters
        used = {e.source for e in edges} | {e.target for e in edges}
        if used:
            nodes = [n for n in nodes if n.id in used]

    return GraphifyGraph(
        nodes=nodes,
        edges=edges,
        directed=graph.directed,
        multigraph=graph.multigraph,
        raw_meta=dict(graph.raw_meta),
    )


def classify_relation_label(confidence: str, confidence_score: Optional[float]) -> str:
    """Map Graphify edge confidence to EXTRACTED / INFERRED / AMBIGUOUS."""
    text = (confidence or "").strip().lower()
    if text in {"extracted", "extract", "explicit", "direct"}:
        return "EXTRACTED"
    if text in {"inferred", "infer", "likely"}:
        return "INFERRED"
    if text in {"ambiguous", "uncertain", "weak"}:
        return "AMBIGUOUS"
    if confidence_score is not None:
        if confidence_score >= 0.8:
            return "EXTRACTED"
        if confidence_score >= 0.5:
            return "INFERRED"
        return "AMBIGUOUS"
    if text:
        return "INFERRED"
    return "AMBIGUOUS"


def compact_graph_context(
    graph: GraphifyGraph,
    *,
    max_chars: int = 4000,
    max_nodes: int = 25,
    max_edges: int = 40,
    question: str = "",
) -> str:
    """Build a compact, labeled graph context block for RAG grounding."""
    lines: List[str] = [
        "### Knowledge Graph Context (Graphify document graph)",
        "Relationships below are machine-extracted from documents.",
        "Labels: EXTRACTED (strong), INFERRED (model-derived), AMBIGUOUS (uncertain).",
        "Do not present INFERRED or AMBIGUOUS edges as verified facts.",
    ]
    if question:
        lines.append(f"Query focus: {question[:200]}")

    nodes = graph.nodes[:max_nodes]
    edges = graph.edges[:max_edges]
    lines.append("")
    lines.append("Nodes:")
    for n in nodes:
        loc = f" @ {n.source_location}" if n.source_location else ""
        src = f" (source: {n.source_file}{loc})" if n.source_file else ""
        lines.append(f"- [{n.id}] {n.label}{src}")

    lines.append("")
    lines.append("Relationships:")
    for e in edges:
        kind = classify_relation_label(e.confidence, e.confidence_score)
        rel = e.relation or "related_to"
        conf = e.confidence or (
            f"{e.confidence_score:.2f}" if e.confidence_score is not None else "n/a"
        )
        lines.append(f"- ({kind}) {e.source} -[{rel}/{conf}]-> {e.target}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n...[truncated]"
    return text


def safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"JSON serialization failed: {exc}")
        return "{}"
