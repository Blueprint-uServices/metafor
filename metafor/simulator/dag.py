from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from metafor.simulator.job import Distribution

if TYPE_CHECKING:
    from metafor.simulator.server import TokenBucket


@dataclass
class NodeConfig:
    """
    Configuration for a single service node in the simulation DAG.

    A NodeConfig contains the static configuration used to construct a
    Server instance. Runtime state, such as the current queue contents,
    active jobs, token count, and timing state, is maintained by the
    corresponding Server and related runtime objects.

    Attributes:
        node_id:
            Unique integer identifier for the node.

        threads:
            Number of worker threads in the node's server thread pool.
            This determines the maximum number of requests that can be
            actively processed concurrently.

        timeout:
            Maximum amount of simulation time allowed for a request attempt
            before the server considers that attempt timed out.

            A timeout may cause a server-side retry when `max_retries`
            permits another attempt.

            Although normally a float, this may also be `None` when no
            server-side timeout is configured.

        max_retries:
            Maximum number of server-side retries permitted for a request
            handled by this node.

        queue_size:
            Maximum number of requests that may wait in the server's queue
            while all worker threads are busy.

            Requests arriving when the queue is full are rejected.

        service_dist:
            Distribution used by the server to sample service/processing
            time for requests handled by this node.

        network_dist:
            Distribution used to sample network delay when communicating
            between this node and other nodes.

            The server may also operate without a network distribution,
            in which case network delay can be treated as zero.

        downstream:
            List of node IDs representing the services called by this
            node.

            An empty list means that this node has no downstream service
            and is therefore a leaf in the service graph.

        token_bucket:
            Optional runtime TokenBucket instance used to perform
            token-bucket admission control for this node.

            When present, a request must successfully consume a token
            before it is admitted to the server. A request is rejected
            when no token is available.

            The TokenBucket itself is configured with:
                - `capacity`: maximum number of tokens that can be stored,
                  which determines the maximum burst size.
                - `refill_rate`: number of tokens added per unit of
                  simulation time, which determines the sustained
                  admission rate.

            The bucket starts with `capacity` tokens. Each admitted request
            consumes one token. Token-bucket admission is independent of
            the server's timeout and retry configuration.

            `None` disables token-bucket admission control.

        name:
            Optional human-readable name for the node.

            When omitted, the name defaults to ``server_<node_id>``.
    """

    node_id: int
    threads: int
    timeout: Optional[float]
    max_retries: int
    queue_size: int
    service_dist: Distribution
    network_dist: Optional[Distribution]
    downstream: list[int] = field(default_factory=list)
    token_bucket: Optional["TokenBucket"] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"server_{self.node_id}"


class DAG:
    """
    Directed acyclic graph of service-node configurations.

    The DAG stores NodeConfig objects indexed by their node ID. Each node's
    `downstream` field defines the outgoing service dependencies.

    Construction validates that every downstream node ID refers to a node
    present in the DAG.
    """

    def __init__(self, nodes: list[NodeConfig]):
        self._nodes: dict[int, NodeConfig] = {n.node_id: n for n in nodes}
        self._validate()

    def _validate(self):
        """
        Validate that every downstream reference points to an existing node.
        """
        all_ids = set(self._nodes)

        for node in self._nodes.values():
            for d in node.downstream:
                if d not in all_ids:
                    raise ValueError(
                        f"Node {node.node_id} references unknown downstream node {d}"
                    )

    def __iter__(self):
        """Iterate over the NodeConfig objects in the DAG."""
        return iter(self._nodes.values())

    def __getitem__(self, node_id: int) -> NodeConfig:
        """Return the NodeConfig associated with `node_id`."""
        return self._nodes[node_id]

    def __contains__(self, node_id: int) -> bool:
        """Return whether `node_id` exists in the DAG."""
        return node_id in self._nodes

    def entry_nodes(self) -> list[NodeConfig]:
        """
        Return nodes that are not downstream of any other node.

        These are the root/entry nodes of the service graph and are the
        nodes to which incoming clients can be attached.
        """
        downstream_ids = {
            d
            for n in self._nodes.values()
            for d in n.downstream
        }

        return [
            n
            for n in self._nodes.values()
            if n.node_id not in downstream_ids
        ]

    def keys(self) -> list[int]:
        """Return the node IDs contained in the DAG."""
        return list(self._nodes.keys())