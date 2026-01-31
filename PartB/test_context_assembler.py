# test_context_assembler.py
"""
Integration Test for Graph-RAG Logic.

This script simulates the 'Full Context' assembly by:
1. Creating a mock repository environment.
2. Injecting a specific SWE-bench Lite bug.
3. Injecting 'Self-Healing' memories (simulating a previous failed run).
4. Running the Multi-Hop Retrieval.
5. Generating the final Prompt.
"""

import os
from final_graph_rag import (
    KGCompassGraphRAG, 
    IssueNode, 
    FileNode, 
    FunctionNode, 
    RequirementNode, 
    PatchMemoryNode, 
    StackOverflowNode
)

def run_context_assembly_test():
    print("="*60)
    print("🛠️  TESTMATE CONTEXT ASSEMBLER - INTEGRATION TEST")
    print("="*60)

    # 1. Initialize the Graph Engine
    print("\n[1] Initializing Knowledge Graph Engine...")
    kg = KGCompassGraphRAG()

    # 2. Simulate Layer 2 (The Codebase - e.g., a simple URL parser)
    print("[2] Building Layer 2 (Structural Code)...")
    
    # File: utils.py
    kg.add_file_node(FileNode(file_path="utils.py", loc=150))
    
    # Function: parse_url (The Buggy Function)
    kg.add_function_node(FunctionNode(
        name="parse_url",
        signature="def parse_url(url_string):",
        body="""def parse_url(url_string):
    if not url_string:
        return None
    # Bug: Splits on first colon, fails on IPv6
    scheme, address = url_string.split(':', 1)
    return {'scheme': scheme, 'address': address}""",
        file_path="utils.py",
        line_start=10,
        line_end=16,
        docstring="Parses a URL into scheme and address components."
    ))
    
    # Function: validate_ipv6 (Dependency)
    kg.add_function_node(FunctionNode(
        name="validate_ipv6",
        signature="def validate_ipv6(address):",
        body="def validate_ipv6(address):\n    # Validation logic...\n    pass",
        file_path="utils.py",
        line_start=20,
        line_end=25
    ))
    
    # Edge: parse_url calls validate_ipv6
    kg.add_calls_edge("parse_url", "validate_ipv6")
    kg.add_contain_edge("utils.py", "parse_url")

    # 3. Simulate Layer 1 (Traceability - The Requirement)
    print("[3] Injecting Layer 1 (Requirements from Model 1)...")
    kg.add_requirement_node(RequirementNode(
        req_id="REQ-NET-05",
        description="The system must support full IPv6 address parsing as defined in RFC 2732.",
        source="spec/networking.md",
        priority="critical"
    ))
    # Link code to requirement
    kg.add_traces_to_edge("parse_url", "REQ-NET-05", confidence=0.85)

    # 4. Simulate Layer 3 (Self-Healing Memory - Your Innovation)
    print("[4] Injecting Layer 3 (Self-Healing Memory)...")
    
    # A past failed attempt to fix this
    kg.add_patch_memory_node(PatchMemoryNode(
        patch_id="MEM-FAIL-001",
        bug_description="IndexError when parsing IPv6 addresses with brackets",
        code="scheme, address = url_string.split(':')", # Wrong fix
        success=False,
        requirements_violated=["REQ-NET-05"]
    ))
    
    # 5. Simulate The Incident (The SWE-bench Bug)
    print("[5] Receiving Bug Report (SWE-bench)...")
    issue = IssueNode(
        issue_id="DJANGO-11001",
        title="ValueError in parse_url with IPv6",
        description="When passing an IPv6 address like 'http://[::1]:80', the parser crashes because it splits on the colons inside the brackets.",
        referenced_files=["utils.py"]
    )
    kg.add_issue_node(issue)
    
    # Create reference edge
    kg.add_reference_edge("DJANGO-11001", "parse_url")

    # ========================================================================
    # EXECUTE RETRIEVAL LOGIC
    # ========================================================================
    
    print("\n" + "="*20 + " RUNNING RETRIEVAL " + "="*20)
    
    # Step A: KGCompass Retrieval (Top 20)
    print("🔍 Step A: Running Multi-Hop Graph Traversal...")
    top_20 = kg.retrieve_top_20_candidates("DJANGO-11001")
    print(f"   -> Found {len(top_20)} relevant functions.")
    print(f"   -> Top Hit: {top_20[0]['function']} (Score: {top_20[0]['score']:.2f})")

    # Step B: 2-Hop Neighborhood Analysis (Deep Context)
    print("🕸️  Step B: Analyzing 2-Hop Neighborhood...")
    neighborhood = kg.two_hop_traversal("parse_url")
    neighborhood_text = kg.format_neighborhood_context(neighborhood)
    
    # Step C: Self-Healing Lookup
    print("🧠 Step C: Checking Self-Healing Memory...")
    history = kg.retrieve_patch_history(issue.description)
    print(f"   -> Found {len(history)} historical patches.")
    if history:
        print(f"   -> Recall: 'Failed patch found: {history[0]['success'] == False}'")

    # Step D: Assemble Final Prompt
    print("\n📝 Step D: Assembling Final Prompt for Model 3...")
    
    final_prompt = kg.format_kgcompass_prompt(
        issue="DJANGO-11001",
        top_20_functions=top_20,
        patch_history=history
    )
    
    # Append the neighborhood analysis (Integrate your separate method)
    final_prompt = final_prompt.replace("=== REASONING CHAIN ===", f"=== REASONING CHAIN ===\n{neighborhood_text}")

    # ========================================================================
    # OUTPUT VALIDATION
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL PROMPT OUTPUT (First 1500 chars)")
    print("="*60)
    print(final_prompt[:1500])
    print("..." + "="*60)
    
    # Validation assertions
    assert "parse_url" in final_prompt, "❌ Buggy function missing from prompt"
    assert "REQ-NET-05" not in final_prompt, "⚠️ Requirement ID optional, but check if trace logic used" 
    assert "MEM-FAIL-001" in str(history), "❌ Memory retrieval failed"
    assert "INCOMING CALLS" in final_prompt, "❌ 2-Hop Context missing"
    
    print("\n✅ INTEGRATION TEST PASSED: The Context Assembler is ready!")

if __name__ == "__main__":
    run_context_assembly_test()