# ============================================================
# FixMate: Mutation Testing Module
# ============================================================
# Verifies test quality by:
#   1. Injecting mutations (bugs) into code
#   2. Running tests against mutated code
#   3. Calculating mutation score (killed/total)
# ============================================================

import ast
import copy
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Mutation:
    """A single mutation applied to code."""
    mutation_id: int
    original: str
    mutated: str
    location: str  # file:line
    operator: str  # Type of mutation
    killed: bool = False
    error: str = ""

@dataclass
class MutationResult:
    """Result of mutation testing."""
    total_mutants: int
    killed: int
    survived: int
    timeout: int
    error: int
    mutation_score: float  # killed / (total - timeout - error)
    mutant_details: List[Mutation] = field(default_factory=list)

# ============================================================
# MUTATION OPERATORS
# ============================================================

class MutationOperator:
    """Base class for mutation operators."""
    name: str = "base"
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        """Generate mutated versions of the node."""
        return []

class ArithmeticOperatorMutation(MutationOperator):
    """Mutate arithmetic operators: + → -, * → /, etc."""
    name = "AOR"
    
    REPLACEMENTS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Div],
        ast.Mult: [ast.Add, ast.Div],
        ast.Div: [ast.Mult, ast.Sub],
        ast.Mod: [ast.Div, ast.Mult],
    }
    
    def mutate(self, node: ast.AST) -> List[Tuple[str, ast.AST]]:
        if not isinstance(node, ast.BinOp):
            return []
        
        op_type = type(node.op)
        if op_type not in self.REPLACEMENTS:
            return []
        
        mutations = []
        for new_op_type in self.REPLACEMENTS[op_type]:
            mutated = copy.deepcopy(node)
            mutated.op = new_op_type()
            mutations.append((f"{op_type.__name__} → {new_op_type.__name__}", mutated))
        
        return mutations

class ComparisonOperatorMutation(MutationOperator):
    """Mutate comparison operators: > → >=, == → !=, etc."""
    name = "COR"
    
    REPLACEMENTS = {
        ast.Gt: [ast.GtE, ast.Lt],
        ast.Lt: [ast.LtE, ast.Gt],
        ast.GtE: [ast.Gt, ast.Lt],
        ast.LtE: [ast.Lt, ast.Gt],
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq],
    }
    
    def mutate(self, node: ast.AST) -> List[Tuple[str, ast.AST]]:
        if not isinstance(node, ast.Compare):
            return []
        
        if len(node.ops) != 1:
            return []
        
        op_type = type(node.ops[0])
        if op_type not in self.REPLACEMENTS:
            return []
        
        mutations = []
        for new_op_type in self.REPLACEMENTS[op_type]:
            mutated = copy.deepcopy(node)
            mutated.ops = [new_op_type()]
            mutations.append((f"{op_type.__name__} → {new_op_type.__name__}", mutated))
        
        return mutations

class BooleanOperatorMutation(MutationOperator):
    """Mutate boolean operators: and → or, True → False."""
    name = "BOR"
    
    def mutate(self, node: ast.AST) -> List[Tuple[str, ast.AST]]:
        mutations = []
        
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                mutated = copy.deepcopy(node)
                mutated.op = ast.Or()
                mutations.append(("And → Or", mutated))
            elif isinstance(node.op, ast.Or):
                mutated = copy.deepcopy(node)
                mutated.op = ast.And()
                mutations.append(("Or → And", mutated))
        
        elif isinstance(node, ast.Constant):
            if node.value is True:
                mutated = copy.deepcopy(node)
                mutated.value = False
                mutations.append(("True → False", mutated))
            elif node.value is False:
                mutated = copy.deepcopy(node)
                mutated.value = True
                mutations.append(("False → True", mutated))
        
        return mutations

class ReturnValueMutation(MutationOperator):
    """Mutate return values: return x → return None."""
    name = "RVM"
    
    def mutate(self, node: ast.AST) -> List[Tuple[str, ast.AST]]:
        if not isinstance(node, ast.Return):
            return []
        
        if node.value is None:
            return []
        
        mutations = []
        
        # Return None instead
        mutated = copy.deepcopy(node)
        mutated.value = ast.Constant(value=None)
        mutations.append(("return X → return None", mutated))
        
        return mutations

# ============================================================
# MUTATION ENGINE
# ============================================================

class MutationEngine:
    """Generate and test mutations."""
    
    def __init__(self):
        self.operators = [
            ArithmeticOperatorMutation(),
            ComparisonOperatorMutation(),
            BooleanOperatorMutation(),
            ReturnValueMutation(),
        ]
    
    def generate_mutants(self, source_code: str, max_mutants: int = 50) -> List[Tuple[str, str, str]]:
        """
        Generate mutated versions of source code.
        
        Returns: List of (mutation_description, mutated_code, location)
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        
        mutants = []
        lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if len(mutants) >= max_mutants:
                break
            
            for operator in self.operators:
                mutations = operator.mutate(node)
                for desc, mutated_node in mutations:
                    if len(mutants) >= max_mutants:
                        break
                    
                    # Get location
                    line_no = getattr(node, 'lineno', 0)
                    location = f"line {line_no}"
                    
                    # Generate mutated code
                    try:
                        mutated_tree = copy.deepcopy(tree)
                        # Replace node in tree
                        mutated_code = self._apply_mutation(source_code, node, mutated_node)
                        if mutated_code and mutated_code != source_code:
                            mutants.append((f"{operator.name}: {desc}", mutated_code, location))
                    except:
                        pass
        
        return mutants
    
    def _apply_mutation(self, source: str, original: ast.AST, mutated: ast.AST) -> str:
        """Apply a single mutation to source code."""
        # Simple text replacement based on line/column
        try:
            lines = source.split('\n')
            line_no = getattr(original, 'lineno', 0) - 1
            
            if 0 <= line_no < len(lines):
                original_text = ast.unparse(original) if hasattr(ast, 'unparse') else str(original)
                mutated_text = ast.unparse(mutated) if hasattr(ast, 'unparse') else str(mutated)
                
                if original_text in lines[line_no]:
                    lines[line_no] = lines[line_no].replace(original_text, mutated_text, 1)
                    return '\n'.join(lines)
        except:
            pass
        
        return None
    
    def run_mutation_testing(self, source_code: str, test_code: str, 
                             function_name: str = "test_func") -> MutationResult:
        """
        Run mutation testing.
        
        1. Generate mutants
        2. Run tests against each mutant
        3. Calculate mutation score
        """
        mutants = self.generate_mutants(source_code, max_mutants=20)
        
        if not mutants:
            return MutationResult(
                total_mutants=0, killed=0, survived=0, timeout=0, error=0,
                mutation_score=0.0
            )
        
        results = []
        killed = 0
        survived = 0
        timeout_count = 0
        error_count = 0
        
        for i, (desc, mutated_code, location) in enumerate(mutants):
            mutation = Mutation(
                mutation_id=i,
                original=desc.split(" → ")[0] if " → " in desc else desc,
                mutated=desc.split(" → ")[1] if " → " in desc else desc,
                location=location,
                operator=desc.split(":")[0] if ":" in desc else "unknown"
            )
            
            # Run test against mutant
            is_killed, error = self._run_test_against_mutant(mutated_code, test_code)
            
            mutation.killed = is_killed
            mutation.error = error
            
            if error == "timeout":
                timeout_count += 1
            elif error:
                error_count += 1
            elif is_killed:
                killed += 1
            else:
                survived += 1
            
            results.append(mutation)
        
        # Calculate score (excluding timeout and errors)
        valid_mutants = len(mutants) - timeout_count - error_count
        score = killed / valid_mutants if valid_mutants > 0 else 0.0
        
        return MutationResult(
            total_mutants=len(mutants),
            killed=killed,
            survived=survived,
            timeout=timeout_count,
            error=error_count,
            mutation_score=score,
            mutant_details=results
        )
    
    def _run_test_against_mutant(self, mutated_code: str, test_code: str, 
                                  timeout: int = 5) -> Tuple[bool, str]:
        """
        Run test against mutated code.
        
        Returns: (killed, error_message)
        - killed=True means test detected the mutation (GOOD)
        - killed=False means mutation survived (BAD test)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write mutated source
            source_file = Path(tmpdir) / "source.py"
            source_file.write_text(mutated_code)
            
            # Write test file
            test_file = Path(tmpdir) / "test_mutant.py"
            full_test = f"from source import *\n\n{test_code}"
            test_file.write_text(full_test)
            
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", str(test_file), "-x", "-q"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                # If test failed (non-zero exit), mutation was killed
                if result.returncode != 0:
                    return True, ""
                else:
                    return False, ""
                    
            except subprocess.TimeoutExpired:
                return False, "timeout"
            except Exception as e:
                return False, str(e)

# ============================================================
# CLI / DEMO
# ============================================================

def demo():
    """Demo mutation testing."""
    print("="*60)
    print("FixMate: Mutation Testing Module")
    print("="*60)
    
    # Sample source code
    source_code = '''
def calculate_discount(price, quantity):
    """Calculate discount based on quantity."""
    if price <= 0:
        return 0
    
    if quantity >= 10:
        discount = 0.2
    elif quantity >= 5:
        discount = 0.1
    else:
        discount = 0.0
    
    return price * (1 - discount)
'''
    
    # Sample test code
    test_code = '''
def test_basic():
    assert calculate_discount(100, 10) == 80

def test_no_discount():
    assert calculate_discount(100, 1) == 100

def test_zero_price():
    assert calculate_discount(0, 10) == 0
'''
    
    print("\n📝 Source Code:")
    print(source_code)
    
    print("\n🧪 Test Code:")
    print(test_code)
    
    # Generate mutants
    engine = MutationEngine()
    print("\n🔀 Generating Mutants...")
    mutants = engine.generate_mutants(source_code, max_mutants=10)
    
    print(f"\nGenerated {len(mutants)} mutants:")
    for i, (desc, _, location) in enumerate(mutants[:5]):
        print(f"  {i+1}. {desc} at {location}")
    
    if len(mutants) > 5:
        print(f"  ... and {len(mutants) - 5} more")
    
    # Run mutation testing (simulated for demo)
    print("\n📊 Mutation Testing (simulated):")
    print(f"  Total Mutants: {len(mutants)}")
    print(f"  Killed: {int(len(mutants) * 0.7)}")
    print(f"  Survived: {int(len(mutants) * 0.3)}")
    print(f"  Mutation Score: 70%")
    
    print("\n✅ A higher mutation score means better tests!")

if __name__ == "__main__":
    demo()
