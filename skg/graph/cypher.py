"""Cypher query templates for motif search."""

# Motif patterns for scenario search
MOTIFS = {
    "cut_in_then_brake": """
    MATCH (o:Agent)-[c:CUTS_IN]->(e:Ego)-[b:DECELERATES]->()
    WHERE c.valid_to <= b.valid_from
    RETURN o, c, e, b
    """,
    "unprotected_left_turn": """
    MATCH (e:Ego)-[t:TRAVELS_ON_LANE]->(l:Lane{turn: 'left'})
    MATCH (v:Agent)-[y:YIELDS_TO]->(e)
    WHERE y.valid_from >= t.valid_from AND y.valid_to <= t.valid_to
    RETURN e, t, l, v, y
    """,
    "occluded_crossing": """
    MATCH (p:Agent{type: 'PEDESTRIAN'})<-[:OCCLUDES]-(v:Agent)
    MATCH (v)-[c:CROSSES_PATH_OF]->(e:Ego)
    RETURN p, v, c, e
    """,
    "near_miss": """
    MATCH (a:Agent)-[nm:NEAR_MISS]->(b:Agent)
    WHERE nm.min_ttc < 3.0
    RETURN a, nm, b
    """,
    "lead_vehicle_braking": """
    MATCH (lead:Agent)-[f:FOLLOWS]->(ego:Ego)
    WHERE f.min_gap < 20.0 AND f.min_ttc < 5.0
    RETURN lead, f, ego
    """,
}


def get_motif_query(motif_name: str) -> str:
    """Get a Cypher query template for a motif."""
    return MOTIFS.get(motif_name, "")


def get_all_motifs() -> list[str]:
    """Get all available motif names."""
    return list(MOTIFS.keys())
