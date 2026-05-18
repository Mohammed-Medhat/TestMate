# ============================================================
# visualize_paths.py - CFG Visualization for Seminar Demo
# ============================================================
"""
Generates a visual Control Flow Graph (CFG) to demonstrate 
FixMate's "Depth" analysis for the seminar presentation.

Usage:
    python visualize_paths.py

Generates: fixmate_cfg.png
"""

import ast
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def create_cfg_visual(code_snippet: str, output_file: str = "fixmate_cfg.png", title: str = None):
    """
    Generates a visual Control Flow Graph (CFG) from Python code.
    This shows exactly what FixMate's CFGPathExtractor "sees" internally.
    """
    tree = ast.parse(code_snippet)
    func = tree.body[0]  # Assume first node is function
    func_name = func.name if hasattr(func, 'name') else "Function"
    
    G = nx.DiGraph()
    G.add_node("START", color='green')
    
    last_node = "START"
    branch_count = 0
    node_colors = {"START": "lightgreen"}
    
    def process_statements(statements, parent_node):
        nonlocal branch_count
        current = parent_node
        
        for node in statements:
            if isinstance(node, ast.If):
                # Branching logic - Decision diamond
                try:
                    cond = ast.unparse(node.test)[:30]  # Limit length
                except:
                    cond = "condition"
                    
                decision_node = f"IF ({cond})?"
                G.add_node(decision_node)
                node_colors[decision_node] = "gold"
                G.add_edge(current, decision_node)
                
                # True branch
                true_label = f"TRUE_{branch_count}"
                if node.body and isinstance(node.body[0], ast.Raise):
                    try:
                        exc = ast.unparse(node.body[0].exc)[:20]
                    except:
                        exc = "Error"
                    true_label = f"RAISE {exc}"
                    node_colors[true_label] = "salmon"
                elif node.body and isinstance(node.body[0], ast.Return):
                    try:
                        val = ast.unparse(node.body[0].value)[:20]
                    except:
                        val = "value"
                    true_label = f"RETURN {val}"
                    node_colors[true_label] = "lightgreen"
                else:
                    node_colors[true_label] = "lightblue"
                    
                G.add_node(true_label)
                G.add_edge(decision_node, true_label, label="True")
                
                # False branch (else or continue)
                if node.orelse:
                    false_label = f"ELSE_{branch_count}"
                    node_colors[false_label] = "lightgrey"
                    G.add_node(false_label)
                    G.add_edge(decision_node, false_label, label="False")
                    current = false_label
                else:
                    current = decision_node  # Continue from decision
                    
                branch_count += 1
                
            elif isinstance(node, ast.Raise):
                try:
                    exc = ast.unparse(node.exc)[:25]
                except:
                    exc = "Exception"
                end_node = f"RAISE {exc}"
                G.add_node(end_node)
                node_colors[end_node] = "salmon"
                G.add_edge(current, end_node)
                current = None
                
            elif isinstance(node, ast.Return):
                try:
                    val = ast.unparse(node.value)[:25] if node.value else "None"
                except:
                    val = "value"
                end_node = f"RETURN {val}"
                G.add_node(end_node)
                node_colors[end_node] = "lightgreen"
                G.add_edge(current, end_node)
                current = None
                
            elif isinstance(node, ast.For):
                loop_node = f"FOR loop_{branch_count}"
                G.add_node(loop_node)
                node_colors[loop_node] = "lightyellow"
                G.add_edge(current, loop_node)
                current = loop_node
                branch_count += 1
                
        return current
    
    process_statements(func.body, "START")

    # Draw the graph
    plt.figure(figsize=(12, 8))
    
    # Use hierarchical layout for better visualization
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.circular_layout(G)
    
    # Get colors for nodes
    colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f"FixMate CFG Analysis: {func_name}()", fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Generated CFG Visualization: {output_file}")
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return output_file


# ============================================================
# Example Usage: Generate slide-ready images
# ============================================================

if __name__ == "__main__":
    # Example 1: Simple function with validation
    sample_code_1 = """
def process_payment(amount, currency):
    if amount < 0:
        raise ValueError("Negative amount")
    if currency != "USD":
        raise ValueError("Invalid currency")
    return True
"""
    create_cfg_visual(sample_code_1, "cfg_payment.png", 
                      "FixMate: Execution Path Analysis")

    # Example 2: Loop with conditional
    sample_code_2 = """
def validate_items(items):
    if not items:
        raise ValueError("Empty list")
    for item in items:
        if item.price <= 0:
            raise ValueError("Invalid price")
    return len(items)
"""
    create_cfg_visual(sample_code_2, "cfg_validate.png",
                      "FixMate: Complex Path Detection")

    print("\n🎯 Images ready for seminar slides!")
    print("   - cfg_payment.png: Payment validation CFG")
    print("   - cfg_validate.png: Item validation CFG")
