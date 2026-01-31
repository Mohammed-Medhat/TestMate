# test_two_hop.py - Test the 2-Hop Traversal Query
"""Tests the KGCompass 2-Hop Traversal implementation."""

from final_graph_rag import KGCompassGraphRAG, FunctionNode, FileNode

def test_two_hop_traversal():
    # Create a test graph
    graph = KGCompassGraphRAG()

    # Add file
    graph.add_file_node(FileNode(file_path='utils.py', language='python'))

    # Add functions
    funcs = [
        FunctionNode(name='main', signature='def main():', body='return extract_data()', file_path='utils.py', line_start=1, line_end=3),
        FunctionNode(name='extract_data', signature='def extract_data():', body='return process(data)', file_path='utils.py', line_start=5, line_end=10),
        FunctionNode(name='process', signature='def process(data):', body='return data.strip()', file_path='utils.py', line_start=12, line_end=15),
        FunctionNode(name='validate', signature='def validate(x):', body='return bool(x)', file_path='utils.py', line_start=17, line_end=19),
    ]
    for f in funcs:
        graph.add_function_node(f)

    # Add edges
    graph.add_contain_edge('utils.py', 'main')
    graph.add_contain_edge('utils.py', 'extract_data')
    graph.add_contain_edge('utils.py', 'process')
    graph.add_contain_edge('utils.py', 'validate')
    graph.add_calls_edge('main', 'extract_data')
    graph.add_calls_edge('extract_data', 'process')
    graph.add_calls_edge('extract_data', 'validate')

    # Test 2-hop traversal
    print('Testing 2-Hop Traversal Query on seed function: extract_data')
    print('='*60)

    neighborhood = graph.two_hop_traversal('extract_data')

    print(f"Seed: {neighborhood['seed']['name']}")
    print(f"Callers: {[c['name'] for c in neighborhood['callers']]}")
    print(f"Callees: {[c['name'] for c in neighborhood['callees']]}")
    print(f"Parent: {neighborhood['parent']['name'] if neighborhood['parent'] else None}")
    print(f"Siblings: {[s['name'] for s in neighborhood['siblings']]}")
    print()
    print('Formatted Context:')
    print('-'*60)
    print(graph.format_neighborhood_context(neighborhood))

if __name__ == "__main__":
    test_two_hop_traversal()
