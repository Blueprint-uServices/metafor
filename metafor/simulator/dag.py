from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NodeConfig:
    node_id:      int
    threads:      int
    timeout:      float
    max_retries:  int
    queue_size:   int
    service_dist: object
    network_dist: object
    downstream:   list[int]          = field(default_factory=list)
    token_bucket: Optional[object]   = None
    name:         Optional[str]      = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"server_{self.node_id}"


class DAG:
    def __init__(self, nodes: list[NodeConfig]):
        self._nodes: dict[int, NodeConfig] = {n.node_id: n for n in nodes}
        self._validate()

    def _validate(self):
        all_ids = set(self._nodes)
        for node in self._nodes.values():
            for d in node.downstream:
                if d not in all_ids:
                    raise ValueError(
                        f"Node {node.node_id} references unknown downstream node {d}"
                    )

    def __iter__(self):
        return iter(self._nodes.values())

    def __getitem__(self, node_id: int) -> NodeConfig:
        return self._nodes[node_id]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self._nodes

    def entry_nodes(self) -> list[NodeConfig]:
        """Nodes that are not downstream of any other node (i.e. roots)."""
        downstream_ids = {d for n in self._nodes.values() for d in n.downstream}
        return [n for n in self._nodes.values() if n.node_id not in downstream_ids]

    def keys(self) -> list[int]:
        return list(self._nodes.keys())