from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import maximum_spanning_arborescence


def max_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX max_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result