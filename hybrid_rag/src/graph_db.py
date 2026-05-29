import json

class CustomGraphDB:
    """
    A lightweight, pure-Python Graph Database implementation.
    Used as an offline fallback when NetworkX is not available.
    Supports nodes, edges, properties, and basic traversals.
    """
    def __init__(self):
        self.nodes = {} # node_id -> attributes dict
        self.edges = {} # node_id -> list of (neighbor_id, rel_type)
        
    def add_node(self, node_id, node_type, **kwargs):
        self.nodes[node_id] = {"type": node_type, **kwargs}
        if node_id not in self.edges:
            self.edges[node_id] = []
            
    def add_edge(self, source, target, rel_type):
        if source in self.nodes and target in self.nodes:
            # Directed edge
            self.edges[source].append((target, rel_type))
            # Treat as undirected for easy retrieval walk
            if target not in self.edges:
                self.edges[target] = []
            self.edges[target].append((source, rel_type + "_OF"))
            
    def get_neighbors(self, node_id):
        return self.edges.get(node_id, [])
        
    def find_nodes_by_type(self, node_type):
        return [node_id for node_id, attrs in self.nodes.items() if attrs.get("type") == node_type]

class KnowledgeGraphManager:
    """
    Manages the Physics Knowledge Graph.
    Uses NetworkX when available, and falls back to CustomGraphDB.
    """
    def __init__(self):
        self.use_networkx = False
        try:
            import networkx as nx
            self.use_networkx = True
            self.graph = nx.Graph()
            print("NetworkX Knowledge Graph engine initialized.")
        except ImportError:
            self.graph = CustomGraphDB()
            print("Using local CustomGraphDB engine (NetworkX not available).")
            
    def add_node(self, node_id, node_type, **kwargs):
        if self.use_networkx:
            self.graph.add_node(node_id, type=node_type, **kwargs)
        else:
            self.graph.add_node(node_id, node_type, **kwargs)
            
    def add_edge(self, source, target, rel_type):
        if self.use_networkx:
            self.graph.add_edge(source, target, relation=rel_type)
        else:
            self.graph.add_edge(source, target, rel_type)
            
    def get_node_attributes(self, node_id):
        if self.use_networkx:
            return self.graph.nodes[node_id] if node_id in self.graph else None
        else:
            return self.graph.nodes.get(node_id, None)
            
    def search_graph(self, query, max_depth=2):
        """
        Token matches query words against concept/topic/formula node names,
        then traverses the graph to collect related formulas, chapters, and pages.
        Returns: dict containing retrieved concepts, formulas, pages, and traversal steps.
        """
        # Filter out generic stop words and search commands to avoid over-matching in graph traversal
        generic_tokens = {'show', 'find', 'explain', 'what', 'page', 'formula', 'chapter', 'topic', 'concept', 'theory', 'law', 'state', 'write', 'give', 'define', 'compare', 'and', 'the', 'its', 'their', 'with'}
        query_tokens = [w.lower() for w in re.findall(r'[a-z0-9]+', query)]
        query_tokens = [t for t in query_tokens if t not in generic_tokens and len(t) > 2]
        matched_nodes = []
        
        # 1. Match seed nodes (nodes that contain query tokens in their name)
        all_nodes = list(self.graph.nodes.keys()) if self.use_networkx else list(self.graph.nodes.keys())
        for node in all_nodes:
            node_lower = str(node).lower()
            # If a query word matches a node title, mark it as seed
            for token in query_tokens:
                if token in node_lower:
                    matched_nodes.append(node)
                    break
                    
        # 2. Traverse Graph from seed nodes to collect context
        retrieved_context = {
            "concepts": set(),
            "formulas": set(),
            "pages": set(),
            "chapters": set(),
            "steps": []
        }
        
        visited = set()
        queue = [(node, 0) for node in matched_nodes]
        
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            
            # Get node attributes
            attrs = self.get_node_attributes(node)
            node_type = attrs.get("type") if attrs else "Unknown"
            
            # Map node to context categories
            if node_type == "Concept" or node_type == "Topic":
                retrieved_context["concepts"].add(node)
            elif node_type == "Formula":
                formula_val = attrs.get("value", node)
                retrieved_context["formulas"].add(f"{node}: {formula_val}")
            elif node_type == "Page":
                retrieved_context["pages"].add(attrs.get("number", node))
            elif node_type == "Chapter":
                retrieved_context["chapters"].add(node)
                
            # Log traversal step
            parent = "Start" if depth == 0 else "Previous"
            retrieved_context["steps"].append(f"Depth {depth}: Visited '{node}' ({node_type})")
            
            # Fetch neighbors
            if self.use_networkx:
                neighbors = list(self.graph.neighbors(node))
                for n in neighbors:
                    queue.append((n, depth + 1))
            else:
                neighbors = self.graph.get_neighbors(node)
                for n, _ in neighbors:
                    queue.append((n, depth + 1))
                    
        # Convert sets to sorted lists for JSON serialization
        return {
            "concepts": sorted(list(retrieved_context["concepts"])),
            "formulas": sorted(list(retrieved_context["formulas"])),
            "pages": sorted(list(retrieved_context["pages"])),
            "chapters": sorted(list(retrieved_context["chapters"])),
            "steps": retrieved_context["steps"]
        }

import re

def build_physics_knowledge_graph():
    """
    Constructs the static seed knowledge graph of chapters, topics, formulas,
    and pages matching NCERT Physics Part 1.
    """
    kg = KnowledgeGraphManager()
    
    # 1. Add Chapters
    chapters = [
        "Chapter 1: Electric Charges and Fields",
        "Chapter 2: Electrostatic Potential and Capacitance",
        "Chapter 3: Current Electricity",
        "Chapter 4: Moving Charges and Magnetism",
        "Chapter 6: Electromagnetic Induction"
    ]
    for ch in chapters:
        kg.add_node(ch, "Chapter")
        
    # 2. Add Topics, Concepts, Formulas, and Pages
    # Coulomb's Law
    kg.add_node("Coulomb's Law", "Topic")
    kg.add_node("Electrostatic Force", "Concept")
    kg.add_node("Coulomb Formula", "Formula", value="F = (1 / (4 * pi * epsilon_0)) * (q1 * q2 / r^2)")
    kg.add_node("Page 9", "Page", number=9)
    # Link Coulomb
    kg.add_edge("Chapter 1: Electric Charges and Fields", "Coulomb's Law", "HAS_TOPIC")
    kg.add_edge("Coulomb's Law", "Electrostatic Force", "EXPLAINS_CONCEPT")
    kg.add_edge("Electrostatic Force", "Coulomb Formula", "REQUIRES_FORMULA")
    kg.add_edge("Coulomb's Law", "Page 9", "DISCUSSED_ON")
    
    # Electric Flux
    kg.add_node("Electric Flux", "Topic")
    kg.add_node("Surface Integral", "Concept")
    kg.add_node("Flux Formula", "Formula", value="Phi_E = E * A * cos(theta)")
    kg.add_node("Page 25", "Page", number=25)
    # Link Flux
    kg.add_edge("Chapter 1: Electric Charges and Fields", "Electric Flux", "HAS_TOPIC")
    kg.add_edge("Electric Flux", "Surface Integral", "EXPLAINS_CONCEPT")
    kg.add_edge("Surface Integral", "Flux Formula", "REQUIRES_FORMULA")
    kg.add_edge("Electric Flux", "Page 25", "DISCUSSED_ON")
    
    # Gauss's Law
    kg.add_node("Gauss's Law", "Topic")
    kg.add_node("Enclosed Charge", "Concept")
    kg.add_node("Gauss Formula", "Formula", value="oint E . dA = q / epsilon_0")
    kg.add_node("Page 33", "Page", number=33)
    # Link Gauss
    kg.add_edge("Chapter 1: Electric Charges and Fields", "Gauss's Law", "HAS_TOPIC")
    kg.add_edge("Gauss's Law", "Enclosed Charge", "EXPLAINS_CONCEPT")
    kg.add_edge("Enclosed Charge", "Gauss Formula", "REQUIRES_FORMULA")
    kg.add_edge("Gauss's Law", "Page 33", "DISCUSSED_ON")
    kg.add_edge("Gauss's Law", "Electric Flux", "RELATED_TO") # Cross-reference
    
    # Electrostatic Potential
    kg.add_node("Electrostatic Potential", "Topic")
    kg.add_node("Potential Formula", "Formula", value="V = q / (4 * pi * epsilon_0 * r)")
    kg.add_node("Page 53", "Page", number=53)
    # Link Potential
    kg.add_edge("Chapter 2: Electrostatic Potential and Capacitance", "Electrostatic Potential", "HAS_TOPIC")
    kg.add_edge("Electrostatic Potential", "Potential Formula", "REQUIRES_FORMULA")
    kg.add_edge("Electrostatic Potential", "Page 53", "DISCUSSED_ON")
    kg.add_edge("Electrostatic Potential", "Electrostatic Force", "RELATED_TO") # Relates back to Chapter 1 force
    
    # Capacitance
    kg.add_node("Capacitors and Capacitance", "Topic")
    kg.add_node("Capacitance Definition", "Concept")
    kg.add_node("Capacitance Basic Formula", "Formula", value="C = Q / V")
    kg.add_node("Parallel Plate Formula", "Formula", value="C = epsilon_0 * A / d")
    kg.add_node("Page 74", "Page", number=74)
    # Link Capacitance
    kg.add_edge("Chapter 2: Electrostatic Potential and Capacitance", "Capacitors and Capacitance", "HAS_TOPIC")
    kg.add_edge("Capacitors and Capacitance", "Capacitance Definition", "EXPLAINS_CONCEPT")
    kg.add_edge("Capacitance Definition", "Capacitance Basic Formula", "REQUIRES_FORMULA")
    kg.add_edge("Capacitance Definition", "Parallel Plate Formula", "REQUIRES_FORMULA")
    kg.add_edge("Capacitors and Capacitance", "Page 74", "DISCUSSED_ON")
    
    # Ohm's Law
    kg.add_node("Ohm's Law", "Topic")
    kg.add_node("Resistance", "Concept")
    kg.add_node("Ohm Formula", "Formula", value="V = I * R")
    kg.add_node("Resistance Dimensions Formula", "Formula", value="R = rho * l / A")
    kg.add_node("Page 95", "Page", number=95)
    # Link Ohm
    kg.add_edge("Chapter 3: Current Electricity", "Ohm's Law", "HAS_TOPIC")
    kg.add_edge("Ohm's Law", "Resistance", "EXPLAINS_CONCEPT")
    kg.add_edge("Resistance", "Ohm Formula", "REQUIRES_FORMULA")
    kg.add_edge("Resistance", "Resistance Dimensions Formula", "REQUIRES_FORMULA")
    kg.add_edge("Ohm's Law", "Page 95", "DISCUSSED_ON")
    
    # Biot-Savart Law
    kg.add_node("Biot-Savart Law", "Topic")
    kg.add_node("Magnetic Field Element", "Concept")
    kg.add_node("Biot-Savart Formula", "Formula", value="dB = (mu_0 / 4 * pi) * (I * dl * sin(theta) / r^2)")
    kg.add_node("Page 143", "Page", number=143)
    # Link Biot-Savart
    kg.add_edge("Chapter 4: Moving Charges and Magnetism", "Biot-Savart Law", "HAS_TOPIC")
    kg.add_edge("Biot-Savart Law", "Magnetic Field Element", "EXPLAINS_CONCEPT")
    kg.add_edge("Magnetic Field Element", "Biot-Savart Formula", "REQUIRES_FORMULA")
    kg.add_edge("Biot-Savart Law", "Page 143", "DISCUSSED_ON")
    
    # Ampere's Circuital Law
    kg.add_node("Ampere's Circuital Law", "Topic")
    kg.add_node("Magnetic Closed Line Integral", "Concept")
    kg.add_node("Ampere Formula", "Formula", value="oint B . dl = mu_0 * I")
    kg.add_node("Page 147", "Page", number=147)
    # Link Ampere
    kg.add_edge("Chapter 4: Moving Charges and Magnetism", "Ampere's Circuital Law", "HAS_TOPIC")
    kg.add_edge("Ampere's Circuital Law", "Magnetic Closed Line Integral", "EXPLAINS_CONCEPT")
    kg.add_edge("Magnetic Closed Line Integral", "Ampere Formula", "REQUIRES_FORMULA")
    kg.add_edge("Ampere's Circuital Law", "Page 147", "DISCUSSED_ON")
    kg.add_edge("Ampere's Circuital Law", "Biot-Savart Law", "RELATED_TO") # Cross-reference
    
    # Faraday's Law
    kg.add_node("Faraday's Law of Induction", "Topic")
    kg.add_node("Electromagnetic Induction", "Concept")
    kg.add_node("Induced EMF", "Concept")
    kg.add_node("Faraday Formula", "Formula", value="epsilon = - d(Phi_B) / dt")
    kg.add_node("Page 207", "Page", number=207)
    # Link Faraday
    kg.add_edge("Chapter 6: Electromagnetic Induction", "Faraday's Law of Induction", "HAS_TOPIC")
    kg.add_edge("Faraday's Law of Induction", "Electromagnetic Induction", "EXPLAINS_CONCEPT")
    kg.add_edge("Electromagnetic Induction", "Induced EMF", "RELATED_TO")
    kg.add_edge("Induced EMF", "Faraday Formula", "REQUIRES_FORMULA")
    kg.add_edge("Faraday's Law of Induction", "Page 207", "DISCUSSED_ON")
    
    # Lenz's Law
    kg.add_node("Lenz's Law", "Topic")
    kg.add_node("Energy Conservation", "Concept")
    kg.add_node("Page 210", "Page", number=210)
    # Link Lenz
    kg.add_edge("Chapter 6: Electromagnetic Induction", "Lenz's Law", "HAS_TOPIC")
    kg.add_edge("Lenz's Law", "Energy Conservation", "EXPLAINS_CONCEPT")
    kg.add_edge("Lenz's Law", "Faraday's Law of Induction", "RELATED_TO") # Links back to Faraday
    kg.add_edge("Lenz's Law", "Page 210", "DISCUSSED_ON")
    
    return kg

if __name__ == '__main__':
    kg = build_physics_knowledge_graph()
    query = "Show me Coulomb's Law formula and its page."
    print(f"Query: '{query}'")
    context = kg.search_graph(query)
    print(f"Graph Search Context:\n{json.dumps(context, indent=2)}")
