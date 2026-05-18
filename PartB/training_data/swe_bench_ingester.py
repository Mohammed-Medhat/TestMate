# swe_bench_ingester.py - Real Data Ingestion from SWE-bench
"""
Ingests real SWE-bench bugs into the Graph-RAG engine.

Pipeline:
1. Load SWE-bench Train from disk
2. Clone repo at specific commit
3. Parse code with KGCompassParser
4. Create IssueNode from problem_statement
5. Build knowledge graph

Usage:
    python swe_bench_ingester.py --dry-run django__django-11001
    python swe_bench_ingester.py --ingest-all --limit 10
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional
from datasets import load_from_disk

from ast_parser_complete import KGCompassParser
from final_graph_rag import KGCompassGraphRAG, IssueNode

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/swe_bench_train")
REPOS_DIR = Path("repos")
GRAPHS_DIR = Path("graphs")

# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

class RepoManager:
    """Handles git operations for SWE-bench repositories."""
    
    def __init__(self, repos_dir: Path = REPOS_DIR):
        self.repos_dir = repos_dir
        self.repos_dir.mkdir(exist_ok=True)
    
    def get_repo_path(self, repo_name: str) -> Path:
        """Get local path for a repository."""
        # e.g., "django/django" -> "repos/django__django"
        safe_name = repo_name.replace("/", "__")
        return self.repos_dir / safe_name
    
    def clone_or_update(self, repo_name: str, commit: str) -> Path:
        """
        Clone repository and checkout specific commit.
        
        Args:
            repo_name: GitHub repo (e.g., "django/django")
            commit: Git commit hash to checkout
            
        Returns:
            Path to cloned repository
        """
        repo_path = self.get_repo_path(repo_name)
        github_url = f"https://github.com/{repo_name}.git"
        
        if not repo_path.exists():
            print(f"  📥 Cloning {repo_name}...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", github_url, str(repo_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try full clone if shallow clone fails
                subprocess.run(
                    ["git", "clone", github_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
        
        # Fetch the specific commit
        print(f"  🔄 Checking out commit {commit[:8]}...")
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        return repo_path
    
    def cleanup(self, repo_name: str):
        """Remove cloned repository to save space."""
        repo_path = self.get_repo_path(repo_name)
        if repo_path.exists():
            shutil.rmtree(repo_path)

# ============================================================================
# SWE-BENCH INGESTER
# ============================================================================

class SWEBenchIngester:
    """
    Ingests SWE-bench bugs into the Graph-RAG engine.
    
    Connects the real dataset to your tested engine!
    """
    
    def __init__(self):
        self.dataset = None
        self.repo_manager = RepoManager()
        GRAPHS_DIR.mkdir(exist_ok=True)
    
    def load_dataset(self):
        """Load SWE-bench Train from disk."""
        print("📂 Loading SWE-bench Train dataset...")
        self.dataset = load_from_disk(str(DATA_DIR))
        print(f"   → Loaded {len(self.dataset)} instances")
        return self.dataset
    
    def find_instance(self, instance_id: str) -> Optional[Dict]:
        """Find a specific instance by ID."""
        if self.dataset is None:
            self.load_dataset()
        
        for i, instance in enumerate(self.dataset):
            if instance['instance_id'] == instance_id:
                return dict(instance)
        return None
    
    def extract_buggy_files_from_patch(self, patch: str) -> list:
        """Extract file paths from a git diff patch."""
        files = []
        for line in patch.split('\n'):
            if line.startswith('diff --git'):
                # Extract: diff --git a/path/to/file.py b/path/to/file.py
                match = re.search(r'a/(.+?) b/', line)
                if match:
                    files.append(match.group(1))
        return list(set(files))
    
    def ingest_single_bug(self, instance: Dict, skip_clone: bool = False) -> KGCompassGraphRAG:
        """
        Ingest a single SWE-bench bug into Graph-RAG.
        
        Args:
            instance: SWE-bench instance dict
            skip_clone: If True, assumes repo already cloned
            
        Returns:
            KGCompassGraphRAG with the bug's knowledge graph
        """
        instance_id = instance['instance_id']
        repo_name = instance['repo']
        base_commit = instance['base_commit']
        problem_statement = instance['problem_statement']
        patch = instance['patch']
        
        print(f"\n{'='*60}")
        print(f"🔧 Ingesting: {instance_id}")
        print(f"   Repo: {repo_name}")
        print(f"   Commit: {base_commit[:8]}")
        print(f"{'='*60}")
        
        # Step 1: Clone/checkout repository
        if not skip_clone:
            repo_path = self.repo_manager.clone_or_update(repo_name, base_commit)
        else:
            repo_path = self.repo_manager.get_repo_path(repo_name)
        
        # Step 2: Parse repository with KGCompassParser
        print("  🔍 Parsing repository...")
        parser = KGCompassParser()
        
        # Exclude common non-code directories
        exclude_dirs = ['venv', '.git', '__pycache__', 'node_modules', 
                       'docs', 'tests', 'test', 'examples', 'benchmarks',
                       '.tox', '.eggs', 'build', 'dist']
        
        graph = parser.parse_repository(str(repo_path), exclude_dirs=exclude_dirs)
        
        # Step 3: Create IssueNode from problem_statement
        print("  📝 Creating Issue node...")
        buggy_files = self.extract_buggy_files_from_patch(patch)
        
        issue = IssueNode(
            issue_id=instance_id,
            title=problem_statement[:100] + "..." if len(problem_statement) > 100 else problem_statement,
            description=problem_statement,
            referenced_files=buggy_files
        )
        graph.add_issue_node(issue)
        
        # Step 4: Create REFERENCE edges from issue to buggy files
        for file_path in buggy_files:
            # Try to find matching file in graph
            for node in graph.graph.nodes():
                if node.endswith(file_path) or file_path in node:
                    graph.add_reference_edge(instance_id, node)
                    break
        
        print(f"  ✅ Graph built!")
        print(f"     → Files: {graph.stats.get('file', 0)}")
        print(f"     → Classes: {graph.stats.get('class', 0)}")
        print(f"     → Functions: {graph.stats.get('function', 0)}")
        print(f"     → Buggy files: {buggy_files}")
        
        return graph
    
    def run_dry_run(self, instance_id: str):
        """
        Run full pipeline on a single bug for validation.
        
        This is your test before processing all 19K bugs!
        """
        print("="*60)
        print("🧪 DRY RUN: Full Pipeline Validation")
        print("="*60)
        
        # Find the instance
        instance = self.find_instance(instance_id)
        if not instance:
            print(f"❌ Instance '{instance_id}' not found in dataset!")
            print("   Available repos:", set(i['repo'] for i in list(self.dataset)[:100]))
            return None
        
        # Ingest the bug
        graph = self.ingest_single_bug(instance)
        
        # Test retrieval
        print("\n" + "="*40)
        print("🔍 Testing Retrieval...")
        print("="*40)
        
        try:
            top_20 = graph.retrieve_top_20_candidates(instance_id, max_candidates=10)
            print(f"\n📊 Top 10 Candidate Functions:")
            for i, candidate in enumerate(top_20[:10], 1):
                print(f"   {i}. {candidate['function']} (Score: {candidate['score']:.3f})")
        except Exception as e:
            print(f"⚠️  Retrieval failed: {e}")
        
        # Test 2-hop traversal on first candidate
        if top_20:
            print("\n" + "="*40)
            print("🕸️  Testing 2-Hop Traversal...")
            print("="*40)
            
            try:
                seed_func = top_20[0]['function']
                neighborhood = graph.two_hop_traversal(seed_func)
                context = graph.format_neighborhood_context(neighborhood)
                print(context[:1000])
            except Exception as e:
                print(f"⚠️  2-Hop failed: {e}")
        
        # Generate prompt
        print("\n" + "="*40)
        print("📝 Generated Prompt (Preview):")
        print("="*40)
        
        try:
            prompt = graph.format_kgcompass_prompt(instance_id, top_20)
            print(prompt[:1500])
            print("...[truncated]")
        except Exception as e:
            print(f"⚠️  Prompt generation failed: {e}")
        
        # Save graph
        graph_path = GRAPHS_DIR / f"{instance_id}.pkl"
        graph.save(str(graph_path))
        print(f"\n💾 Graph saved to: {graph_path}")
        
        print("\n" + "="*60)
        print("✅ DRY RUN COMPLETE!")
        print("="*60)
        
        return graph
    
    def ingest_batch(self, limit: int = None, repo_filter: str = None):
        """
        Ingest multiple bugs (for training data generation).
        
        Args:
            limit: Max bugs to process
            repo_filter: Only process bugs from this repo (e.g., "django/django")
        """
        if self.dataset is None:
            self.load_dataset()
        
        processed = 0
        errors = 0
        
        for instance in self.dataset:
            if limit and processed >= limit:
                break
            
            if repo_filter and instance['repo'] != repo_filter:
                continue
            
            try:
                graph = self.ingest_single_bug(dict(instance))
                graph_path = GRAPHS_DIR / f"{instance['instance_id']}.pkl"
                graph.save(str(graph_path))
                processed += 1
                
                # Cleanup to save disk space (optional)
                # self.repo_manager.cleanup(instance['repo'])
                
            except Exception as e:
                print(f"❌ Error processing {instance['instance_id']}: {e}")
                errors += 1
        
        print(f"\n{'='*60}")
        print(f"📊 BATCH COMPLETE: {processed} processed, {errors} errors")
        print(f"{'='*60}")

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest SWE-bench into Graph-RAG")
    parser.add_argument("--dry-run", type=str, help="Run dry-run on specific instance ID")
    parser.add_argument("--ingest-all", action="store_true", help="Ingest all instances")
    parser.add_argument("--limit", type=int, default=10, help="Limit for batch ingestion")
    parser.add_argument("--repo", type=str, help="Filter by repository")
    parser.add_argument("--list", action="store_true", help="List available instances")
    
    args = parser.parse_args()
    
    ingester = SWEBenchIngester()
    
    if args.list:
        ingester.load_dataset()
        repos = set(i['repo'] for i in ingester.dataset)
        print(f"\n📂 Available repositories ({len(repos)}):")
        for repo in sorted(repos)[:20]:
            count = sum(1 for i in ingester.dataset if i['repo'] == repo)
            print(f"   {repo}: {count} instances")
        print(f"\n   ...and {len(repos) - 20} more")
    
    elif args.dry_run:
        ingester.run_dry_run(args.dry_run)
    
    elif args.ingest_all:
        ingester.ingest_batch(limit=args.limit, repo_filter=args.repo)
    
    else:
        # Default: show dataset info
        ingester.load_dataset()
        print("\nUse --dry-run <instance_id> to test with a specific bug")
        print("Use --list to see available instances")

if __name__ == "__main__":
    main()
