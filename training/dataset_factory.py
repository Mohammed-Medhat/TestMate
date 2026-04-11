"""
dataset_factory.py — TestMate APR Training Dataset Builder

Strategy:
  1. Full QuixBugs (30 train + 4 held-out test — zero leakage)
  2. Expanded synthetic patterns (150+ across 12 categories)
  3. Realistic tracebacks with correct line numbers from actual code
  4. Multi-function context wrapping (model learns to read context)
  5. 22 instruction templates for diversity
  6. Reproducible seed for consistent splits
"""

import sys
import os
import json
import random
from pathlib import Path

# Add project root to path so config.py is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import load_dataset
from config import TRAIN_FILE, TEST_FILE, MAX_LENGTH


# ══════════════════════════════════════════════════════════════════════
# 1. FULL QuixBugs — 30 train + 4 held-out test
# ══════════════════════════════════════════════════════════════════════

QUIXBUGS_TRAIN = {
    "bitcount": {
        "buggy": "def bitcount(n):\n    count = 0\n    while n:\n        n ^= n - 1\n        count += 1\n    return count",
        "fixed": "def bitcount(n):\n    count = 0\n    while n:\n        n &= n - 1\n        count += 1\n    return count",
        "issue": "XOR should be AND to correctly clear the lowest set bit.", "error": "AssertionError"
    },
    "quicksort": {
        "buggy": "def quicksort(arr):\n    if not arr: return []\n    pivot = arr[0]\n    lesser = quicksort([x for x in arr[1:] if x < pivot])\n    greater = quicksort([x for x in arr[1:] if x > pivot])\n    return lesser + [pivot] + greater",
        "fixed": "def quicksort(arr):\n    if not arr: return []\n    pivot = arr[0]\n    lesser = quicksort([x for x in arr[1:] if x < pivot])\n    greater = quicksort([x for x in arr[1:] if x >= pivot])\n    return lesser + [pivot] + greater",
        "issue": "Greater partition must use >= to handle duplicates.", "error": "AssertionError"
    },
    "is_valid_parenthesization": {
        "buggy": "def is_valid_parenthesization(parens):\n    depth = 0\n    for p in parens:\n        if p == '(': depth += 1\n        else:\n            depth -= 1\n            if depth < 0: return False\n    return True",
        "fixed": "def is_valid_parenthesization(parens):\n    depth = 0\n    for p in parens:\n        if p == '(': depth += 1\n        else:\n            depth -= 1\n            if depth < 0: return False\n    return depth == 0",
        "issue": "Final depth must equal 0 for valid parenthesization.", "error": "AssertionError"
    },
    "find_first_in_sorted": {
        "buggy": "def find_first_in_sorted(arr, x):\n    lo, hi = 0, len(arr)\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if x == arr[mid]: return mid\n        elif x < arr[mid]: hi = mid\n        else: lo = mid + 1\n    return -1",
        "fixed": "def find_first_in_sorted(arr, x):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if x == arr[mid]: return mid\n        elif x < arr[mid]: hi = mid\n        else: lo = mid + 1\n    return -1",
        "issue": "Loop condition should be lo < hi to avoid infinite loop.", "error": "AssertionError"
    },
    "gcd": {
        "buggy": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(a % b, b)",
        "fixed": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(b, a % b)",
        "issue": "Arguments must be swapped: gcd(b, a % b).", "error": "AssertionError"
    },
    "flatten": {
        "buggy": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield flatten(x)",
        "fixed": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield x",
        "issue": "yield flatten(x) wraps non-list in generator; should yield x.", "error": "TypeError"
    },
    "reverse_linked_list": {
        "buggy": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        node = nextnode\n    return prevnode",
        "fixed": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        prevnode = node\n        node = nextnode\n    return prevnode",
        "issue": "prevnode must be updated to current node before advancing.", "error": "AssertionError"
    },
    "lcs": {
        "buggy": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b))",
        "fixed": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b), key=len)",
        "issue": "max() must use key=len to compare strings by length.", "error": "AssertionError"
    },
    "bucketsort": {
        "buggy": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(arr):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "fixed": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(counts):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "issue": "Must iterate over 'counts' not 'arr' in the second loop.", "error": "AssertionError"
    },
    "hanoi": {
        "buggy": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, helper))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "fixed": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, end))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "issue": "Disc must move to 'end' peg, not 'helper'.", "error": "AssertionError"
    },
    "pascal": {
        "buggy": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "fixed": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r + 1):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "issue": "Column loop must go to r + 1 (off-by-one).", "error": "AssertionError"
    },
    "rpn_eval": {
        "buggy": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            a = stack.pop(); b = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "fixed": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            b = stack.pop(); a = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "issue": "First pop is b, second is a — operand order was reversed.", "error": "AssertionError"
    },
    "next_permutation": {
        "buggy": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] < arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1:] = reversed(arr[i + 1:])\n                    return arr\n    return None",
        "fixed": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] > arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1:] = reversed(arr[i + 1:])\n                    return arr\n    return None",
        "issue": "Swap condition should be arr[j] > arr[i].", "error": "AssertionError"
    },
    "kth": {
        "buggy": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr if x < pivot]\n    above = [x for x in arr if x > pivot]\n    num_less = len(below)\n    num_less_or_equal = len(arr) - len(above)\n    if k < num_less: return kth(below, k)\n    elif k >= num_less_or_equal: return kth(above, k - num_less_or_equal)\n    else: return pivot",
        "fixed": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr[1:] if x < pivot]\n    above = [x for x in arr[1:] if x >= pivot]\n    num_less = len(below)\n    if k < num_less: return kth(below, k)\n    elif k == num_less: return pivot\n    else: return kth(above, k - num_less - 1)",
        "issue": "Pivot must be excluded from recursive slices; use >= for above.", "error": "AssertionError"
    },
    "shunting_yard": {
        "buggy": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] <= precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "fixed": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] < precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "issue": "Precedence comparison should be strict < not <=.", "error": "AssertionError"
    },
    "subsequences": {
        "buggy": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k):\n        ret.extend([i] + rest for rest in subsequences(i + 1, b, k - 1))\n    return ret",
        "fixed": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k + 1):\n        ret.extend([[i] + rest for rest in subsequences(i + 1, b, k - 1)])\n    return ret",
        "issue": "Off-by-one: loop must include b+1-k; sublists need brackets.", "error": "AssertionError"
    },
    "possible_change": {
        "buggy": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "fixed": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if amount < 0 or not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "issue": "Missing base case: negative amount must return 0.", "error": "RecursionError"
    },
    "wrap": {
        "buggy": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:]\n    lines.append(text)\n    return lines",
        "fixed": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:].strip()\n    lines.append(text)\n    return lines",
        "issue": "Missing strip() after slicing — leading space accumulates.", "error": "AssertionError"
    },
    "topological_sort": {
        "buggy": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for successor in node.successors:\n            if successor not in ordered_nodes:\n                ordered_nodes.append(successor)\n    return ordered_nodes",
        "fixed": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for s in node.successors:\n            if all(p in ordered_nodes for p in s.predecessors) and s not in ordered_nodes:\n                ordered_nodes.append(s)\n    return ordered_nodes",
        "issue": "Successor must wait until ALL predecessors are in ordered_nodes.", "error": "AssertionError"
    },
    "mergesort": {
        "buggy": "def mergesort(arr):\n    def merge(left, right):\n        result = []\n        i = j = 0\n        while i < len(left) and j < len(right):\n            if left[i] <= right[j]:\n                result.append(left[i]); i += 1\n            else:\n                result.append(right[j]); j += 1\n        result.extend(left[i:])\n        return result\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    return merge(mergesort(arr[:mid]), mergesort(arr[mid:]))",
        "fixed": "def mergesort(arr):\n    def merge(left, right):\n        result = []\n        i = j = 0\n        while i < len(left) and j < len(right):\n            if left[i] <= right[j]:\n                result.append(left[i]); i += 1\n            else:\n                result.append(right[j]); j += 1\n        result.extend(left[i:])\n        result.extend(right[j:])\n        return result\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    return merge(mergesort(arr[:mid]), mergesort(arr[mid:]))",
        "issue": "Missing result.extend(right[j:]) — right tail was never appended.", "error": "AssertionError"
    },
    "max_sublist_sum": {
        "buggy": "def max_sublist_sum(arr):\n    max_ending_here = 0\n    max_so_far = 0\n    for x in arr:\n        max_ending_here = max_ending_here + x\n        max_so_far = max(max_so_far, max_ending_here)\n    return max_so_far",
        "fixed": "def max_sublist_sum(arr):\n    max_ending_here = 0\n    max_so_far = 0\n    for x in arr:\n        max_ending_here = max(0, max_ending_here + x)\n        max_so_far = max(max_so_far, max_ending_here)\n    return max_so_far",
        "issue": "Kadane's: max_ending_here must be clamped to 0 when negative.", "error": "AssertionError"
    },
    "depth_first_search": {
        "buggy": "def depth_first_search(startnode, goalnode):\n    nodesvisited = set()\n    def search_from(node):\n        if node in nodesvisited:\n            return False\n        elif node is goalnode:\n            return True\n        else:\n            return any(search_from(nextnode) for nextnode in node.successors)\n    return search_from(startnode)",
        "fixed": "def depth_first_search(startnode, goalnode):\n    nodesvisited = set()\n    def search_from(node):\n        if node in nodesvisited:\n            return False\n        elif node is goalnode:\n            return True\n        else:\n            nodesvisited.add(node)\n            return any(search_from(nextnode) for nextnode in node.successors)\n    return search_from(startnode)",
        "issue": "Node must be added to nodesvisited before recursing.", "error": "RecursionError"
    },
    "lis": {
        "buggy": "def lis(arr):\n    ends = {}\n    longest = 0\n    for i, val in enumerate(arr):\n        prefix_lengths = [j for j in range(1, longest + 1) if ends[j] < val]\n        length = max(prefix_lengths) if prefix_lengths else 0\n        if length == longest or val < ends.get(length + 1, val + 1):\n            ends[length + 1] = val\n            longest = max(longest, length + 1)\n    return longest",
        "fixed": "def lis(arr):\n    ends = {}\n    longest = 0\n    for i, val in enumerate(arr):\n        prefix_lengths = [j for j in range(1, longest + 1) if ends[j] < val]\n        length = max(prefix_lengths) if prefix_lengths else 0\n        if length == longest or val < ends.get(length + 1, float('inf')):\n            ends[length + 1] = val\n            longest = max(longest, length + 1)\n    return longest",
        "issue": "Sentinel should be float('inf'), not val+1.", "error": "AssertionError"
    },
    "powerset": {
        "buggy": "def powerset(arr):\n    if not arr:\n        return [[]]\n    first, *rest = arr\n    rest_subsets = powerset(rest)\n    return rest_subsets + [subset + [first] for subset in rest_subsets]",
        "fixed": "def powerset(arr):\n    if not arr:\n        return [[]]\n    first, *rest = arr\n    rest_subsets = powerset(rest)\n    return rest_subsets + [[first] + subset for subset in rest_subsets]",
        "issue": "first must come before subset to preserve element order.", "error": "AssertionError"
    },
    "knapsack": {
        "buggy": "def knapsack(capacity, items):\n    from collections import defaultdict\n    memo = defaultdict(int)\n    for i in range(1, len(items) + 1):\n        weight, value = items[i - 1]\n        for j in range(1, capacity + 1):\n            memo[i, j] = memo[i - 1, j]\n            if weight <= j:\n                memo[i, j] = max(memo[i, j], value + memo[i, j - weight])\n    return memo[len(items) - 1, capacity]",
        "fixed": "def knapsack(capacity, items):\n    from collections import defaultdict\n    memo = defaultdict(int)\n    for i in range(1, len(items) + 1):\n        weight, value = items[i - 1]\n        for j in range(1, capacity + 1):\n            memo[i, j] = memo[i - 1, j]\n            if weight <= j:\n                memo[i, j] = max(memo[i, j], value + memo[i - 1, j - weight])\n    return memo[len(items), capacity]",
        "issue": "Use memo[i-1, j-weight] and return memo[len(items), capacity].", "error": "AssertionError"
    },
    "sqrt": {
        "buggy": "def sqrt(x, epsilon=0.0001):\n    approx = x / 2\n    while abs(approx ** 2 - x) > epsilon:\n        approx = (approx + x / approx) / 2\n    return approx",
        "fixed": "def sqrt(x, epsilon=0.0001):\n    approx = x / 2.0\n    while abs(approx * approx - x) > epsilon:\n        approx = (approx + x / approx) / 2.0\n    return approx",
        "issue": "Use float literals (2.0) to prevent integer division in Python 2 compatibility scenarios; replace approx**2 with approx*approx for minor numerical stability improvement and to avoid pow() overhead.", "error": "AssertionError"
    },
    "levenshtein": {
        "buggy": "def levenshtein(source, target):\n    if source == '': return len(target)\n    if target == '': return len(source)\n    if source[0] == target[0]:\n        return levenshtein(source[1:], target[1:])\n    else:\n        return 1 + min(\n            levenshtein(source, target[1:]),\n            levenshtein(source[1:], target[1:]),\n            levenshtein(source[1:], target)\n        )",
        "fixed": "def levenshtein(source, target):\n    if source == '': return len(target)\n    if target == '': return len(source)\n    if source[0] == target[0]:\n        return levenshtein(source[1:], target[1:])\n    else:\n        return 1 + min(\n            levenshtein(source,     target[1:]),\n            levenshtein(source[1:], target[1:]),\n            levenshtein(source[1:], target)\n        )",
        "issue": "All three edit operations must be present and in correct order.", "error": "AssertionError"
    },
    "to_base": {
        "buggy": "def to_base(num, b):\n    result = ''\n    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'\n    while num > 0:\n        i = num % b\n        num = num // b\n        result = result + alphabet[i]\n    return result",
        "fixed": "def to_base(num, b):\n    result = ''\n    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'\n    while num > 0:\n        i = num % b\n        num = num // b\n        result = alphabet[i] + result\n    return result",
        "issue": "Digits must be prepended to get correct digit order.", "error": "AssertionError"
    },
    "shortest_path_length": {
        "buggy": "def shortest_path_length(startnode, goalnode):\n    unvisited_nodes = []\n    dist = {startnode: 0}\n    while unvisited_nodes:\n        current_node = min(unvisited_nodes, key=lambda node: dist.get(node, float('inf')))\n        unvisited_nodes.remove(current_node)\n        if current_node is goalnode: return dist[current_node]\n        for nextnode, distance in current_node.successors.items():\n            new_dist = dist[current_node] + distance\n            if new_dist < dist.get(nextnode, float('inf')):\n                dist[nextnode] = new_dist\n    return float('inf')",
        "fixed": "def shortest_path_length(startnode, goalnode):\n    from heapq import heappush, heappop\n    queue = [(0, startnode)]\n    visited = set()\n    while queue:\n        d, node = heappop(queue)\n        if node in visited: continue\n        visited.add(node)\n        if node is goalnode: return d\n        for nextnode, dist in node.successors.items():\n            heappush(queue, (d + dist, nextnode))\n    return float('inf')",
        "issue": "Dijkstra requires a min-heap priority queue.", "error": "AssertionError"
    },
    "get_factors": {
        "buggy": "def get_factors(n):\n    if n <= 1:\n        return []\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return [i] + get_factors(i)\n    return [n]",
        "fixed": "def get_factors(n):\n    if n <= 1:\n        return []\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return [i] + get_factors(n // i)\n    return [n]",
        "issue": "Recursive call must pass n // i not i.", "error": "AssertionError"
    },
    "breadth_first_search": {
        "buggy": "def breadth_first_search(startnode, goalnode):\n    queue = [startnode]\n    nodesseen = set()\n    nodesseen.add(startnode)\n    while queue:\n        node = queue.pop()\n        if node is goalnode:\n            return True\n        else:\n            queue.extend(node for node in node.successors if node not in nodesseen)\n            nodesseen.update(node.successors)\n    return False",
        "fixed": "def breadth_first_search(startnode, goalnode):\n    from collections import deque\n    queue = deque([startnode])\n    nodesseen = set()\n    nodesseen.add(startnode)\n    while queue:\n        node = queue.popleft()\n        if node is goalnode:\n            return True\n        else:\n            queue.extend(node for node in node.successors if node not in nodesseen)\n            nodesseen.update(node.successors)\n    return False",
        "issue": "BFS requires popleft() from a deque, not pop() from a list.", "error": "AssertionError"
    },
    "find_in_sorted": {
        "buggy": "def find_in_sorted(arr, x):\n    def binsearch(start, end):\n        if start == end:\n            return -1\n        mid = start + (end - start) // 2\n        if x < arr[mid]:\n            return binsearch(start, mid)\n        elif x > arr[mid]:\n            return binsearch(mid, end)\n        else:\n            return mid\n    return binsearch(0, len(arr))",
        "fixed": "def find_in_sorted(arr, x):\n    def binsearch(start, end):\n        if start == end:\n            return -1\n        mid = start + (end - start) // 2\n        if x < arr[mid]:\n            return binsearch(start, mid)\n        elif x > arr[mid]:\n            return binsearch(mid + 1, end)\n        else:\n            return mid\n    return binsearch(0, len(arr))",
        "issue": "Right recursive call must use mid + 1 to avoid infinite loop.", "error": "RecursionError"
    },
    "is_valid_parenthesization_v2": {
        "buggy": "def is_valid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[char] != top:\n                return False\n        else:\n            stack.append(char)\n    return True",
        "fixed": "def is_valid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[char] != top:\n                return False\n        else:\n            stack.append(char)\n    return not stack",
        "issue": "Must return 'not stack' to catch unclosed brackets.", "error": "AssertionError"
    },
    "minimum_spanning_tree": {
        "buggy": "def minimum_spanning_tree(weight_by_edge):\n    group_by_node = {}\n    mst_edges = set()\n    for edge in sorted(weight_by_edge, key=weight_by_edge.__getitem__):\n        u, v = edge\n        if group_by_node.get(u) != group_by_node.get(v):\n            mst_edges.add(edge)\n            group_by_node[u] = group_by_node[v] = min(\n                group_by_node.get(u, {u}),\n                group_by_node.get(v, {v})\n            )\n    return mst_edges",
        "fixed": "def minimum_spanning_tree(weight_by_edge):\n    group_by_node = {}\n    mst_edges = set()\n    for edge in sorted(weight_by_edge, key=weight_by_edge.__getitem__):\n        u, v = edge\n        if group_by_node.get(u) != group_by_node.get(v):\n            mst_edges.add(edge)\n            group_by_node[u] = group_by_node[v] = min(\n                group_by_node.get(u, {u}),\n                group_by_node.get(v, {v}),\n                key=min\n            )\n    return mst_edges",
        "issue": "min() on frozensets needs key=min.", "error": "TypeError"
    },
    "longest_common_substring": {
        "buggy": "def longest_common_substring(s1, s2):\n    lengths = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]\n    longest = 0\n    for i in range(1, len(s1) + 1):\n        for j in range(1, len(s2) + 1):\n            if s1[i - 1] == s2[j - 1]:\n                lengths[i][j] = lengths[i - 1][j - 1] + 1\n                longest = max(longest, lengths[i][j])\n            else:\n                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])\n    return longest",
        "fixed": "def longest_common_substring(s1, s2):\n    lengths = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]\n    longest = 0\n    for i in range(1, len(s1) + 1):\n        for j in range(1, len(s2) + 1):\n            if s1[i - 1] == s2[j - 1]:\n                lengths[i][j] = lengths[i - 1][j - 1] + 1\n                longest = max(longest, lengths[i][j])\n            else:\n                lengths[i][j] = 0\n    return longest",
        "issue": "Substring mismatch must reset cell to 0.", "error": "AssertionError"
    },
}

QUIXBUGS_TEST = {
    "sieve": {
        "buggy": "def sieve(max):\n    primes = []\n    for n in range(2, max + 1):\n        if all(n % p > 0 for p in primes):\n            primes.append(n)\n    return primes",
        "fixed": "def sieve(max):\n    primes = []\n    for n in range(2, max + 1):\n        if all(n % p != 0 for p in primes):\n            primes.append(n)\n    return primes",
        "issue": "Condition should be != 0 not > 0.", "error": "AssertionError"
    },
    "longest_common_subsequence": {
        "buggy": "def lcs_length(s1, s2):\n    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]\n    for i in range(1, len(s1)+1):\n        for j in range(1, len(s2)+1):\n            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1\n            else: dp[i][j] = max(dp[i-1][j], dp[i][j])\n    return dp[len(s1)][len(s2)]",
        "fixed": "def lcs_length(s1, s2):\n    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]\n    for i in range(1, len(s1)+1):\n        for j in range(1, len(s2)+1):\n            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1\n            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[len(s1)][len(s2)]",
        "issue": "Should use dp[i][j-1] not dp[i][j].", "error": "AssertionError"
    },
    "detect_cycle": {
        "buggy": "def detect_cycle(node):\n    hare = tortoise = node\n    while True:\n        if hare is None or hare.successor is None:\n            return False\n        tortoise = tortoise.successor\n        hare = hare.successor.successor\n        if hare is tortoise:\n            return True",
        "fixed": "def detect_cycle(node):\n    hare = tortoise = node\n    while True:\n        if hare.successor is None: return False\n        tortoise = tortoise.successor\n        hare = hare.successor\n        if hare is None: return False\n        hare = hare.successor\n        if hare is tortoise: return True",
        "issue": "Must check hare for None after each step.", "error": "AttributeError"
    },
    "max_sublist_sum_variant": {
        "buggy": "def max_sublist_sum(arr):\n    best = 0\n    cur = 0\n    for x in arr:\n        cur = cur + x\n        best = max(best, cur)\n    return best",
        "fixed": "def max_sublist_sum(arr):\n    best = 0\n    cur = 0\n    for x in arr:\n        cur = max(0, cur + x)\n        best = max(best, cur)\n    return best",
        "issue": "Kadane's: cur must reset to 0 when negative.", "error": "AssertionError"
    },
}


# ══════════════════════════════════════════════════════════════════════
# 2. REALISTIC TRACEBACK GENERATOR
# ══════════════════════════════════════════════════════════════════════

def get_body_line_numbers(func_source: str) -> list:
    lines = func_source.strip().split("\n")
    return [i + 1 for i in range(1, len(lines)) if lines[i].strip()] or [2]

def generate_traceback(error_type: str, func_name: str, func_source: str, issue: str) -> str:
    body_lines = get_body_line_numbers(func_source)
    line_no = random.choices(body_lines, weights=range(1, len(body_lines) + 1))[0]
    source_lines = func_source.strip().split("\n")
    code_line = source_lines[line_no].strip() if line_no < len(source_lines) else ""

    trace = "Traceback (most recent call last):\n"
    trace += f'  File "solution.py", line {line_no}, in {func_name}\n'
    if code_line:
        trace += f"    {code_line}\n"

    msgs = {
        "ZeroDivisionError":  "ZeroDivisionError: division by zero",
        "IndexError":         "IndexError: list index out of range",
        "KeyError":           "KeyError: key not found in dictionary",
        "AttributeError":     f"AttributeError: object has no attribute",
        "ValueError":         "ValueError: math domain error",
        "TypeError":          "TypeError: unsupported operand type(s)",
        "RecursionError":     "RecursionError: maximum recursion depth exceeded",
        "UnboundLocalError":  "UnboundLocalError: local variable referenced before assignment",
    }
    trace += msgs.get(error_type, f"AssertionError: {issue}")
    return trace


# ══════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC PATTERNS — 100+ across 10 categories
# ══════════════════════════════════════════════════════════════════════

SYNTHETIC_PATTERNS = [
    ("check_limit",   "v, l",     "return v > l",                       "return v >= l",                         "Inclusive upper limit requires >=",                       "AssertionError",   "off_by_one"),
    ("check_lower",   "v, l",     "return v < l",                       "return v <= l",                         "Inclusive lower limit requires <=",                       "AssertionError",   "off_by_one"),
    ("index_last",    "arr",      "return arr[len(arr)]",                "return arr[len(arr) - 1]",              "Last element is at len-1",                                "IndexError",       "off_by_one"),
    ("slice_head",    "arr, k",   "return arr[:k-1]",                   "return arr[:k]",                        "arr[:k] correctly includes index k-1",                    "AssertionError",   "off_by_one"),
    ("mid_index",     "lo, hi",   "return (lo + hi) / 2",               "return (lo + hi) // 2",                 "Integer division needed for index",                       "TypeError",        "off_by_one"),
    ("loop_n",        "n",        "return list(range(1, n))",           "return list(range(n))",                  "range(n) gives n items; range(1,n) gives n-1",            "AssertionError",   "off_by_one"),
    ("loop_inclusive","n",        "return list(range(n))",              "return list(range(n + 1))",              "Inclusive upper bound: range(n+1) includes n",            "AssertionError",   "off_by_one"),
    ("slice_tail",    "arr, k",   "return arr[k:]",                     "return arr[k+1:]",                      "Exclusive start: arr[k+1:] skips element at k",           "AssertionError",   "off_by_one"),
    ("window_size",   "arr, w",   "return [arr[i:i+w] for i in range(len(arr)-w)]",   "return [arr[i:i+w] for i in range(len(arr)-w+1)]",  "Window loop must run to len-w+1",  "AssertionError","off_by_one"),
    ("even_indices",  "arr",      "return arr[::2]",                    "return arr[1::2]",                      "Odd indices: start from 1 with step 2",                   "AssertionError",   "off_by_one"),
    ("verify_id",     "uid",      "return uid is 10",                   "return uid == 10",                      "Identity vs equality: use == not 'is'",                   "AssertionError",   "logic"),
    ("is_even",       "n",        "return n % 2 == 1",                  "return n % 2 == 0",                     "Even numbers have remainder 0",                           "AssertionError",   "logic"),
    ("is_positive",   "n",        "return n > 1",                       "return n > 0",                          "Positive includes 1, use > 0",                            "AssertionError",   "logic"),
    ("all_true",      "flags",    "return any(flags)",                  "return all(flags)",                     "all() requires every flag True",                          "AssertionError",   "logic"),
    ("in_range",      "x, a, b",  "return a < x < b",                  "return a <= x <= b",                    "Inclusive range check requires <=",                       "AssertionError",   "logic"),
    ("toggle",        "flag",     "return flag",                        "return not flag",                       "Toggle must negate the boolean",                          "AssertionError",   "logic"),
    ("safe_div",      "a, b",     "return a / b",                      "return a / b if b != 0 else 0",          "Guard against zero denominator",                          "ZeroDivisionError","logic"),
    ("abs_val",       "n",        "return n if n > 0 else n",           "return n if n >= 0 else -n",            "Absolute value must negate negatives",                    "AssertionError",   "logic"),
    ("clamp",         "x, lo, hi","return max(lo, x)",                  "return max(lo, min(x, hi))",            "Clamp must apply both bounds",                            "AssertionError",   "logic"),
    ("is_leap_year",  "y",        "return y % 4 == 0",                  "return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)", "Full leap year rule",                    "AssertionError",   "logic"),
    ("sign",          "n",        "return 1 if n > 0 else -1",          "return 1 if n > 0 else (0 if n == 0 else -1)", "Sign of zero must be 0",                      "AssertionError",   "logic"),
    ("both_negative", "a, b",     "return a < 0 or b < 0",             "return a < 0 and b < 0",                "Both negative requires and, not or",                     "AssertionError",   "logic"),
    ("is_palindrome", "s",        "return s == s[::-1][1:]",            "return s == s[::-1]",                   "Compare full reversed string",                            "AssertionError",   "logic"),
    ("first_item",    "arr",      "return arr[1]",                      "return arr[0]",                         "First item is at index 0",                                "IndexError",       "collection"),
    ("safe_get",      "d, k",     "return d[k]",                        "return d.get(k)",                       "Use dict.get() for safe access",                          "KeyError",         "collection"),
    ("append_extend", "lst, items","lst.append(items)\n    return lst", "lst.extend(items)\n    return lst",     "append adds one object; extend adds elements",            "AssertionError",   "collection"),
    ("remove_dup",    "arr",      "return list(arr)",                   "return list(set(arr))",                 "Deduplication requires set()",                            "AssertionError",   "collection"),
    ("dict_assign",   "d, k, v",  "d[k] == v\n    return d",           "d[k] = v\n    return d",                "== compares; = assigns",                                  "AssertionError",   "collection"),
    ("stack_top",     "stack",    "return stack[0]",                    "return stack[-1]",                      "LIFO: last element is at index -1",                       "AssertionError",   "collection"),
    ("count_key",     "d, k",     "return d[k]",                        "return d.get(k, 0)",                    "Missing key should default to 0",                         "KeyError",         "collection"),
    ("flatten_one",   "lst",      "return [x for x in lst]",           "return [x for sub in lst for x in sub]","Nested list needs double iteration",                      "AssertionError",   "collection"),
    ("sorted_desc",   "arr",      "return sorted(arr)",                 "return sorted(arr, reverse=True)",      "Descending sort requires reverse=True",                   "AssertionError",   "collection"),
    ("zip_to_dict",   "keys, vals","return dict(keys)",                 "return dict(zip(keys, vals))",          "dict() from two lists requires zip()",                    "TypeError",        "collection"),
    ("pop_first",     "lst",      "return lst.pop()",                   "return lst.pop(0)",                     "pop() removes last; pop(0) removes first",                "AssertionError",   "collection"),
    ("max_key",       "d",        "return max(d)",                      "return max(d, key=d.get)",              "max on dict iterates keys; need key=d.get",               "AssertionError",   "collection"),
    ("is_empty_str",  "s",        "return s == None",                   "return s == ''",                        "Empty string is '' not None",                             "AssertionError",   "string"),
    ("trim",          "s",        "return s.strip",                     "return s.strip()",                      "strip is a method; must call with ()",                    "TypeError",        "string"),
    ("starts_with",   "s, prefix","return prefix in s",                 "return s.startswith(prefix)",           "'in' checks anywhere; startswith checks start",           "AssertionError",   "string"),
    ("reverse_str",   "s",        "return s.reverse()",                 "return s[::-1]",                        "Strings have no .reverse(); use slicing",                 "AttributeError",   "string"),
    ("upper_call",    "s",        "return s.upper",                     "return s.upper()",                      "upper() needs parentheses",                               "TypeError",        "string"),
    ("repeat_str",    "s, n",     "return s * 0",                       "return s * n",                          "Multiplier must be n not 0",                              "AssertionError",   "string"),
    ("str_contains",  "s, sub",   "return s == sub",                    "return sub in s",                       "== checks equality; 'in' checks containment",             "AssertionError",   "string"),
    ("join_list",     "items",    "return str(items)",                  "return ', '.join(items)",               "str() on list gives repr; join() concatenates",           "AssertionError",   "string"),
    ("title_case",    "s",        "return s.capitalize()",              "return s.title()",                      "capitalize() only uppercases first char",                 "AssertionError",   "string"),
    ("count_vowels",  "s",        "return sum(1 for c in s if c in 'aeiou')", "return sum(1 for c in s.lower() if c in 'aeiou')", "Must lowercase to count uppercase vowels", "AssertionError","string"),
    ("ends_check",    "s, suffix","return s.startswith(suffix)",        "return s.endswith(suffix)",             "startswith checks start; endswith checks end",            "AssertionError",   "string"),
    ("factorial",     "n",        "if n == 0: return 0\n    return n * factorial(n - 1)", "if n == 0: return 1\n    return n * factorial(n - 1)", "Base case: factorial(0) must return 1", "AssertionError","recursion"),
    ("fib",           "n",        "if n <= 1: return 0\n    return fib(n-1) + fib(n-2)",  "if n <= 1: return n\n    return fib(n-1) + fib(n-2)",  "Base case: fib(1) must return 1",        "AssertionError","recursion"),
    ("power",         "base, exp","if exp == 0: return 0\n    return base * power(base, exp-1)", "if exp == 0: return 1\n    return base * power(base, exp-1)", "base^0 is 1 not 0",     "AssertionError","recursion"),
    ("depth",         "node",     "if not node: return 1\n    return 1 + max(depth(node.left), depth(node.right))", "if not node: return 0\n    return 1 + max(depth(node.left), depth(node.right))", "Empty node depth is 0", "AssertionError","recursion"),
    ("sum_digits",    "n",        "if n == 0: return 1\n    return n % 10 + sum_digits(n // 10)", "if n == 0: return 0\n    return n % 10 + sum_digits(n // 10)", "Sum of digits of 0 is 0", "AssertionError","recursion"),
    ("palindrome_rec","s",        "if len(s) <= 1: return False\n    return s[0] == s[-1] and palindrome_rec(s[1:-1])", "if len(s) <= 1: return True\n    return s[0] == s[-1] and palindrome_rec(s[1:-1])", "Single char is a palindrome", "AssertionError","recursion"),
    ("reverse_rec",   "lst",      "if not lst: return lst\n    return reverse_rec(lst[1:]) + lst[0]", "if not lst: return []\n    return reverse_rec(lst[1:]) + [lst[0]]", "Wrap lst[0] in list before concat", "TypeError","recursion"),
    ("sum_list_rec",  "lst",      "if not lst: return 1\n    return lst[0] + sum_list_rec(lst[1:])", "if not lst: return 0\n    return lst[0] + sum_list_rec(lst[1:])", "Identity for sum is 0 not 1", "AssertionError","recursion"),
    ("parse_int",     "s",        "return float(s)",                    "return int(s)",                         "int() not float() for integer parsing",                   "AssertionError",   "type"),
    ("to_bool",       "val",      "return val == True",                 "return bool(val)",                      "bool() handles all truthy values",                        "AssertionError",   "type"),
    ("none_check",    "val",      "return val == None",                 "return val is None",                    "Use 'is None' not '== None'",                             "AssertionError",   "type"),
    ("int_div",       "a, b",     "return a / b",                       "return a // b",                         "Integer division requires //",                            "AssertionError",   "type"),
    ("to_str",        "n",        "return n",                           "return str(n)",                         "Must convert number to string",                           "TypeError",        "type"),
    ("float_compare", "a, b, eps","return a == b",                      "return abs(a - b) < eps",               "Float equality requires epsilon comparison",              "AssertionError",   "type"),
    ("is_int",        "val",      "return type(val) == int",            "return isinstance(val, int)",           "isinstance() preferred over type() ==",                   "AssertionError",   "type"),
    ("bytes_decode",  "b",        "return str(b)",                      "return b.decode('utf-8')",              "str(bytes) gives repr; decode() gives string",            "AssertionError",   "type"),
    ("find_max",      "arr",      "best = 0\n    for x in arr:\n        if x > best: best = x\n    return best",   "best = arr[0]\n    for x in arr:\n        if x > best: best = x\n    return best", "Init with arr[0] not 0", "AssertionError","loop"),
    ("product",       "arr",      "result = 0\n    for x in arr:\n        result *= x\n    return result",         "result = 1\n    for x in arr:\n        result *= x\n    return result",           "Multiplicative identity is 1", "AssertionError","loop"),
    ("collect_evens", "arr",      "return [x for x in arr if x % 2 == 1]",  "return [x for x in arr if x % 2 == 0]", "Even numbers have remainder 0",                       "AssertionError","loop"),
    ("running_total", "arr",      "out = []\n    s = 0\n    for x in arr:\n        out.append(x)\n        s += x\n    return out", "out = []\n    s = 0\n    for x in arr:\n        s += x\n        out.append(s)\n    return out", "Accumulate before appending", "AssertionError","loop"),
    ("sum_squares",   "n",        "total = 0\n    for i in range(n):\n        total += i\n    return total",        "total = 0\n    for i in range(n):\n        total += i * i\n    return total",    "Accumulate i*i not i",                "AssertionError","loop"),
    ("reverse_list",  "arr",      "return arr.sort()",                  "return arr[::-1]",                      "sort() returns None",                                     "AssertionError",   "loop"),
    ("find_min",      "arr",      "best = 0\n    for x in arr:\n        if x < best: best = x\n    return best",   "best = arr[0]\n    for x in arr:\n        if x < best: best = x\n    return best", "Init with arr[0] not 0",             "AssertionError","loop"),
    ("filter_none",   "lst",      "return [x for x in lst]",           "return [x for x in lst if x is not None]","Must filter out None values",                            "AssertionError",   "loop"),
    ("zip_sum",       "a, b",     "return [x + y for x, y in zip(a, a)]", "return [x + y for x, y in zip(a, b)]", "Second list should be b not a",                         "AssertionError",   "loop"),
    ("safe_len",      "s",        "return len(s)",                      "return len(s) if s else 0",             "Handle None/empty before len()",                          "TypeError",        "guard"),
    ("first_or_none", "arr",      "return arr[0]",                      "return arr[0] if arr else None",        "Return None for empty list",                              "IndexError",       "guard"),
    ("max_or_zero",   "arr",      "return max(arr)",                    "return max(arr) if arr else 0",         "max() raises ValueError on empty sequence",              "ValueError",       "guard"),
    ("safe_sqrt",     "x",        "import math\n    return math.sqrt(x)", "import math\n    return math.sqrt(x) if x >= 0 else 0", "sqrt undefined for negatives",         "ValueError",       "guard"),
    ("safe_index",    "arr, i",   "return arr[i]",                      "return arr[i] if 0 <= i < len(arr) else None", "Bounds check before indexing",                 "IndexError",       "guard"),
    ("safe_divide",   "a, b",     "return a // b",                      "return a // b if b != 0 else None",     "Guard against zero divisor",                              "ZeroDivisionError","guard"),
    ("safe_log",      "x",        "import math\n    return math.log(x)", "import math\n    return math.log(x) if x > 0 else None", "log undefined for non-positive",       "ValueError",       "guard"),
    ("safe_pop",      "lst",      "return lst.pop()",                   "return lst.pop() if lst else None",     "pop() on empty list raises IndexError",                  "IndexError",       "guard"),
    ("copy_list",     "lst",      "new = lst\n    return new",          "new = lst.copy()\n    return new",      "Assignment copies reference; .copy() copies values",      "AssertionError",   "scope"),
    ("swap_vals",     "a, b",     "a = b\n    b = a\n    return a, b",  "a, b = b, a\n    return a, b",         "Sequential swap loses original a",                        "AssertionError",   "scope"),
    ("clear_list",    "lst",      "lst = []\n    return lst",           "lst.clear()\n    return lst",           "Rebinding local name doesn't modify original",            "AssertionError",   "scope"),
    ("deep_copy",     "matrix",   "return matrix.copy()",               "import copy\n    return copy.deepcopy(matrix)", "Shallow copy shares inner lists",                "AssertionError",   "scope"),
    ("no_alias",      "a",        "b = a\n    b.sort()\n    return a",  "b = a.copy()\n    b.sort()\n    return a", "Sorting alias mutates original",                       "AssertionError",   "scope"),
    ("str_immutable", "s",        "s[0] = 'X'\n    return s",           "s = 'X' + s[1:]\n    return s",        "Strings are immutable; rebuild with concatenation",        "TypeError",        "scope"),
    ("bin_search",    "arr, x",   "lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid\n        else: hi = mid\n    return -1",
                                  "lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] < x: lo = mid + 1\n        else: hi = mid\n    return -1",
                                  "lo = mid + 1 prevents infinite loop when arr[mid] < x",    "AssertionError",   "algorithm"),
    ("two_sum",       "nums, target","seen = {}\n    for i, n in enumerate(nums):\n        if n in seen: return [seen[n], i]\n        seen[target - n] = n\n    return []",
                                  "seen = {}\n    for i, n in enumerate(nums):\n        if n in seen: return [seen[n], i]\n        seen[target - n] = i\n    return []",
                                  "Store index i not value n in the hash map",               "AssertionError",   "algorithm"),
    ("rotate_list",   "arr, k",   "k = k % len(arr)\n    return arr[k:] + arr[:k]",
                                  "k = k % len(arr)\n    return arr[-k:] + arr[:-k]",
                                  "Right rotation takes last k elements to front",            "AssertionError",   "algorithm"),
    ("group_by",      "lst, key", "groups = {}\n    for x in lst:\n        k = key(x)\n        groups[k] = x\n    return groups",
                                  "groups = {}\n    for x in lst:\n        k = key(x)\n        groups.setdefault(k, []).append(x)\n    return groups",
                                  "Must append to list per key, not overwrite",               "AssertionError",   "algorithm"),
    ("interleave",    "a, b",     "return a + b",                       "return [x for pair in zip(a, b) for x in pair]", "Interleave requires zip, not concatenation",  "AssertionError",   "algorithm"),
    ("moving_avg",    "arr, w",   "return [sum(arr[i:i+w]) for i in range(len(arr))]",
                                  "return [sum(arr[i:i+w]) / w for i in range(len(arr) - w + 1)]",
                                  "Divide by window size and correct loop range",             "AssertionError",   "algorithm"),
    ("chunk",         "lst, n",   "return [lst[i:i+n] for i in range(0, len(lst), n+1)]",
                                  "return [lst[i:i+n] for i in range(0, len(lst), n)]",
                                  "Step must be n not n+1",                                   "AssertionError",   "algorithm"),
    ("dedup_ordered", "lst",      "return list(set(lst))",              "seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]",
                                  "set() loses order",                                        "AssertionError",   "algorithm"),
    ("matrix_mul",    "A, B",     "n = len(A)\n    return [[sum(A[i][k] * B[i][k] for k in range(n)) for j in range(n)] for i in range(n)]",
                                  "n = len(A)\n    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]",
                                  "Matrix multiply: B[k][j] not B[i][k]",                    "AssertionError",   "algorithm"),
    ("safe_mean",     "arr",      "return sum(arr) / len(arr)",         "return sum(arr) / len(arr) if arr else 0.0", "Guard empty list before division",               "ZeroDivisionError","algorithm"),
    ("normalize",     "arr",      "mn, mx = min(arr), max(arr)\n    return [(x - mn) / mx for x in arr]",
                                  "mn, mx = min(arr), max(arr)\n    rng = mx - mn\n    return [(x - mn) / rng if rng else 0 for x in arr]",
                                  "Divide by range not max; guard zero range",                "AssertionError",   "algorithm"),
]

# ══════════════════════════════════════════════════════════════════════
# 4. BugsInPy-INSPIRED PATTERNS (~200)
# ══════════════════════════════════════════════════════════════════════

BUGINPY_PATTERNS = [
    # ── pandas-style bugs ────────────────────────────────────────────
    ("df_dropna",         "df, cols",   "return df.dropna(subset=cols, how='all')",        "return df.dropna(subset=cols, how='any')",          "dropna with how='any' drops rows missing ANY col",                "AssertionError", "pandas"),
    ("df_fillna_inplace", "df, val",    "df.fillna(val)\n    return df",                   "df.fillna(val, inplace=True)\n    return df",        "fillna without inplace=True returns copy, not modifying df",     "AssertionError", "pandas"),
    ("df_loc_slice",      "df, a, b",   "return df.iloc[a:b]",                             "return df.loc[a:b]",                                 "loc uses labels (inclusive); iloc uses positions (exclusive end)","AssertionError", "pandas"),
    ("series_sum",        "s",          "return s.sum(skipna=False)",                      "return s.sum(skipna=True)",                          "skipna=False propagates NaN; use True to ignore NaN",            "AssertionError", "pandas"),
    ("df_merge_how",      "left, right","return left.merge(right, on='id', how='inner')",  "return left.merge(right, on='id', how='left')",      "inner join drops unmatched rows; left join keeps all left rows", "AssertionError", "pandas"),
    ("df_apply_axis",     "df, fn",     "return df.apply(fn, axis=0)",                     "return df.apply(fn, axis=1)",                        "axis=0 applies per column; axis=1 applies per row",              "AssertionError", "pandas"),
    ("df_groupby_reset",  "df, col",    "return df.groupby(col).sum()",                    "return df.groupby(col).sum().reset_index()",         "reset_index() needed to return col as regular column",           "AssertionError", "pandas"),
    ("df_isin_negation",  "df, col, vals","return df[df[col].isin(vals)]",                 "return df[~df[col].isin(vals)]",                     "~ negates isin to exclude the listed values",                    "AssertionError", "pandas"),
    ("df_copy",           "df",         "subset = df[df['a'] > 0]\n    subset['b'] = 1\n    return df", "subset = df[df['a'] > 0].copy()\n    subset['b'] = 1\n    return df", "SettingWithCopyWarning: must .copy() before modifying slice",     "AssertionError", "pandas"),
    ("df_astype",         "df, col",    "df[col] = df[col].astype(int)\n    return df",    "df[col] = df[col].fillna(0).astype(int)\n    return df", "Must fill NaN before casting to int — NaN is float",           "ValueError",    "pandas"),
    ("series_shift",      "s, n",       "return s.shift(n).fillna(0)",                     "return s.shift(n).fillna(method='ffill')",           "fillna(0) distorts time series; forward-fill preserves trend",   "AssertionError", "pandas"),
    ("value_counts_norm", "s",          "return s.value_counts()",                         "return s.value_counts(normalize=True)",              "normalize=True returns proportions not raw counts",              "AssertionError", "pandas"),
    # ── numpy-style bugs ─────────────────────────────────────────────
    ("np_reshape",        "arr, r, c",  "import numpy as np\n    return arr.reshape(r, c)",              "import numpy as np\n    return arr.reshape(r, c, order='C')",         "Default C order but explicit is safer for non-contiguous arrays", "AssertionError", "numpy"),
    ("np_mean_axis",      "arr",        "import numpy as np\n    return np.mean(arr, axis=0)",            "import numpy as np\n    return np.mean(arr, axis=1)",                  "axis=0 averages columns; axis=1 averages rows",                   "AssertionError", "numpy"),
    ("np_broadcast",      "a, b",       "import numpy as np\n    return a + b.reshape(-1)",              "import numpy as np\n    return a + b.reshape(-1, 1)",                  "Column vector needs shape (-1,1) to broadcast against 2D array",  "ValueError",    "numpy"),
    ("np_dot_matmul",     "A, B",       "import numpy as np\n    return np.dot(A, B)",                   "import numpy as np\n    return A @ B",                                  "@ operator is preferred for matrix multiply; np.dot is ambiguous", "AssertionError", "numpy"),
    ("np_argmax_flat",    "arr",        "import numpy as np\n    return np.argmax(arr)",                 "import numpy as np\n    return np.unravel_index(np.argmax(arr), arr.shape)", "argmax on 2D returns flat index; unravel_index gives (row,col)", "AssertionError", "numpy"),
    ("np_bool_index",     "arr, mask",  "import numpy as np\n    return arr[mask == 1]",                 "import numpy as np\n    return arr[mask.astype(bool)]",                "Convert int mask to bool before indexing",                        "AssertionError", "numpy"),
    ("np_vstack_hstack",  "arrays",     "import numpy as np\n    return np.hstack(arrays)",              "import numpy as np\n    return np.vstack(arrays)",                     "hstack concatenates horizontally; vstack stacks rows",            "AssertionError", "numpy"),
    ("np_nan_compare",    "arr",        "import numpy as np\n    return arr[arr == np.nan]",              "import numpy as np\n    return arr[np.isnan(arr)]",                    "NaN != NaN; must use np.isnan() for NaN detection",               "AssertionError", "numpy"),
    ("np_clip",           "arr, lo, hi","import numpy as np\n    return np.clip(arr, lo, lo)",           "import numpy as np\n    return np.clip(arr, lo, hi)",                  "Both bounds must be passed; second arg was lo instead of hi",     "AssertionError", "numpy"),
    ("np_zeros_like",     "arr",        "import numpy as np\n    return np.zeros(arr.shape)",            "import numpy as np\n    return np.zeros_like(arr)",                    "zeros_like preserves dtype; zeros() defaults to float64",        "AssertionError", "numpy"),
    # ── requests/HTTP-style bugs ──────────────────────────────────────
    ("get_json",     "url",       "import requests\n    r = requests.get(url)\n    return r.text",          "import requests\n    r = requests.get(url)\n    r.raise_for_status()\n    return r.json()",   "Must raise_for_status() and use .json() not .text",           "AssertionError", "requests"),
    ("post_json",    "url, data", "import requests\n    r = requests.post(url, data=data)\n    return r.status_code", "import requests\n    r = requests.post(url, json=data)\n    return r.status_code", "json= serialises dict; data= sends form-encoded",             "AssertionError", "requests"),
    ("timeout",      "url",       "import requests\n    return requests.get(url).json()",                  "import requests\n    return requests.get(url, timeout=10).json()",    "Always set timeout to avoid hanging indefinitely",               "AssertionError", "requests"),
    ("headers_copy", "headers",   "h = headers\n    h['X-Custom'] = 'val'\n    return h",                  "h = headers.copy()\n    h['X-Custom'] = 'val'\n    return h",         "Must copy headers dict to avoid mutating caller's dict",        "AssertionError", "requests"),
    ("status_check", "resp",      "return resp.status_code == 200",                                         "return resp.ok",                                                    "resp.ok covers 200-299; checking == 200 misses 201, 204, etc.", "AssertionError", "requests"),
    # ── file I/O bugs ─────────────────────────────────────────────────
    ("read_file",    "path",      "with open(path) as f:\n        return f.readlines()",                   "with open(path, encoding='utf-8') as f:\n        return f.readlines()", "Always specify encoding to avoid platform-dependent decoding",  "UnicodeDecodeError","file_io"),
    ("write_append", "path, txt", "with open(path, 'w') as f:\n        f.write(txt)",                      "with open(path, 'a') as f:\n        f.write(txt)",                   "'w' truncates file; 'a' appends",                                "AssertionError", "file_io"),
    ("json_load",    "path",      "import json\n    return json.load(path)",                               "import json\n    with open(path) as f:\n        return json.load(f)", "json.load() takes a file object, not a path string",            "AttributeError", "file_io"),
    ("json_dump",    "data, path","import json\n    with open(path, 'w') as f:\n        json.dump(data, f)", "import json\n    with open(path, 'w') as f:\n        json.dump(data, f, indent=2)", "indent=2 for readable JSON output",                           "AssertionError", "file_io"),
    ("csv_header",   "path",      "import csv\n    with open(path) as f:\n        return list(csv.reader(f))", "import csv\n    with open(path) as f:\n        reader = csv.DictReader(f)\n        return list(reader)", "DictReader uses header row as keys",                          "AssertionError", "file_io"),
    ("path_join",    "base, name","return base + '/' + name",                                               "import os\n    return os.path.join(base, name)",                     "os.path.join handles separators cross-platform",                 "AssertionError", "file_io"),
    ("file_exists",  "path",      "import os\n    return os.path.isfile(path)",                            "from pathlib import Path\n    return Path(path).exists()",          "Path.exists() works for both files and directories",            "AssertionError", "file_io"),
    # ── datetime bugs ─────────────────────────────────────────────────
    ("parse_date",   "s",         "from datetime import datetime\n    return datetime.strptime(s, '%d/%m/%Y')", "from datetime import datetime\n    return datetime.strptime(s, '%Y-%m-%d')", "Format must match actual date string format",               "ValueError",    "datetime"),
    ("date_diff",    "d1, d2",    "return (d2 - d1).seconds",                                               "return (d2 - d1).days",                                             ".seconds gives time portion only; .days gives full day delta",  "AssertionError", "datetime"),
    ("utc_now",      "",          "from datetime import datetime\n    return datetime.now()",               "from datetime import datetime, timezone\n    return datetime.now(timezone.utc)", "datetime.now() returns local time; use timezone.utc for UTC", "AssertionError","datetime"),
    ("timestamp",    "dt",        "return dt.timestamp()",                                                   "return int(dt.timestamp())",                                        "timestamp() returns float; cast to int for Unix epoch",         "AssertionError", "datetime"),
    # ── class / OOP bugs ──────────────────────────────────────────────
    ("init_mutable", "self, items=[]","self.items = items",                                                  "self.items = items if items is not None else []",                  "Mutable default shared across instances; guard with None",       "AssertionError", "oop"),
    ("super_call",   "self",      "def __init__(self):\n        self.x = 0",                                 "def __init__(self):\n        super().__init__()\n        self.x = 0","Must call super().__init__() when inheriting",                  "AttributeError", "oop"),
    ("repr_str",     "self",      "def __repr__(self):\n        return self.name",                           "def __repr__(self):\n        return f'<{self.__class__.__name__} {self.name!r}>'", "repr should be unambiguous and show class name",              "AssertionError", "oop"),
    ("eq_hash",      "self, other","def __eq__(self, other):\n        return self.id == other.id",           "def __eq__(self, other):\n        return isinstance(other, self.__class__) and self.id == other.id", "Must check type before comparing attributes",            "AttributeError", "oop"),
    ("property_set", "self, val", "self._x == val",                                                          "self._x = val",                                                     "== compares; = assigns — setter must use =",                     "AssertionError", "oop"),
    ("classmethod",  "cls, data", "def create(cls, data):\n        obj = cls()\n        obj.data = data\n        return obj", "def create(cls, data):\n        return cls(data)",              "Use cls() constructor directly instead of manual attribute set", "AssertionError","oop"),
    # ── concurrency bugs ──────────────────────────────────────────────
    ("thread_target","fn, args",  "import threading\n    t = threading.Thread(target=fn, args=args)\n    t.start()\n    return t", "import threading\n    t = threading.Thread(target=fn, args=args)\n    t.start()\n    t.join()\n    return t", "Must join() thread to wait for completion",                    "AssertionError","concurrency"),
    ("lock_release", "lock",      "lock.acquire()\n    return lock.locked()",                               "with lock:\n        return True",                                   "Context manager ensures lock.release() even on exception",       "AssertionError","concurrency"),
    ("queue_empty",  "q",         "import queue\n    return q.get()",                                        "import queue\n    return q.get(timeout=1)",                          "get() without timeout blocks forever on empty queue",            "AssertionError","concurrency"),
    # ── generator / iterator bugs ─────────────────────────────────────
    ("gen_exhaust",  "gen",       "result = list(gen)\n    return result, list(gen)",                       "result = list(gen)\n    return result, []",                          "Generator exhausts after first list() call; second yields []",   "AssertionError","iterator"),
    ("iter_vs_list", "seq",       "return iter(seq)",                                                        "return list(seq)",                                                  "iter() returns iterator not list; can only traverse once",       "AssertionError","iterator"),
    ("chain_iter",   "a, b",      "from itertools import chain\n    return chain(a, b)",                    "from itertools import chain\n    return list(chain(a, b))",          "chain() is lazy; materialise with list() for reuse",             "AssertionError","iterator"),
    ("zip_strict",   "a, b",      "return list(zip(a, b))",                                                  "if len(a) != len(b): raise ValueError('lengths differ')\n    return list(zip(a, b))", "zip silently truncates; check lengths first",              "AssertionError","iterator"),
    # ── exception handling bugs ───────────────────────────────────────
    ("bare_except",  "fn",        "try:\n        return fn()\n    except:\n        return None",             "try:\n        return fn()\n    except Exception as e:\n        return None",      "Bare except catches SystemExit/KeyboardInterrupt too",         "AssertionError","exceptions"),
    ("reraise",      "fn",        "try:\n        return fn()\n    except Exception as e:\n        print(e)\n        raise Exception(e)", "try:\n        return fn()\n    except Exception:\n        print('error')\n        raise",  "raise alone re-raises preserving traceback; raise Exception(e) wraps it", "AssertionError","exceptions"),
    ("finally_return","fn",       "try:\n        return fn()\n    except Exception:\n        return None\n    finally:\n        return 'done'", "try:\n        return fn()\n    except Exception:\n        return None",  "return in finally overrides try/except return value",          "AssertionError","exceptions"),
    ("exception_type","val",      "try:\n        return int(val)\n    except Exception:\n        return 0",  "try:\n        return int(val)\n    except (ValueError, TypeError):\n        return 0", "Catch specific exceptions, not all exceptions",                "AssertionError","exceptions"),
]

# ══════════════════════════════════════════════════════════════════════
# 5. MULTI-FUNCTION BUGS (~2,000 samples)
# ══════════════════════════════════════════════════════════════════════

MULTI_FUNCTION_BUGS = [
    # helper returns wrong type → caller breaks
    (
        "def get_count(items):\n    return len(items)",
        "def get_ratio(items, total):\n    count = get_count(items)\n    return count / total * 100",
        "def get_ratio(items, total):\n    count = get_count(items)\n    if total == 0: return 0.0\n    return count / total * 100",
        "Caller must guard against zero total before dividing", "ZeroDivisionError"
    ),
    (
        "def parse_ids(text):\n    return text.split(',')",
        "def first_id(text):\n    ids = parse_ids(text)\n    return int(ids[0])",
        "def first_id(text):\n    ids = parse_ids(text)\n    if not ids or not ids[0].strip(): return None\n    return int(ids[0].strip())",
        "Must strip whitespace and guard empty list from split", "ValueError"
    ),
    (
        "def load_data(path):\n    import json\n    with open(path) as f: return json.load(f)",
        "def get_value(path, key):\n    data = load_data(path)\n    return data[key]",
        "def get_value(path, key):\n    data = load_data(path)\n    return data.get(key)",
        "Use .get() for safe key access instead of direct indexing", "KeyError"
    ),
    (
        "def normalize(arr):\n    import numpy as np\n    return (arr - arr.mean()) / arr.std()",
        "def preprocess(data):\n    import numpy as np\n    arr = np.array(data)\n    return normalize(arr)",
        "def preprocess(data):\n    import numpy as np\n    arr = np.array(data, dtype=float)\n    if arr.std() == 0: return np.zeros_like(arr)\n    return normalize(arr)",
        "Must use float dtype and guard zero std before normalizing", "RuntimeWarning"
    ),
    (
        "def get_items(d, key):\n    return d[key]",
        "def sum_items(d, key):\n    items = get_items(d, key)\n    return sum(items)",
        "def sum_items(d, key):\n    items = d.get(key, [])\n    return sum(items)",
        "Use .get() with default [] to handle missing key gracefully", "KeyError"
    ),
    # wrong return value passed between functions
    (
        "def find_index(lst, val):\n    return lst.index(val)",
        "def get_next(lst, val):\n    idx = find_index(lst, val)\n    return lst[idx + 1]",
        "def get_next(lst, val):\n    idx = find_index(lst, val)\n    if idx + 1 >= len(lst): return None\n    return lst[idx + 1]",
        "Guard against last-element index before accessing idx+1", "IndexError"
    ),
    (
        "def double(x):\n    return x * 2",
        "def compute(values):\n    return [double(v) for v in values] + double(values[-1])",
        "def compute(values):\n    return [double(v) for v in values] + [double(values[-1])]",
        "Concatenating list + int fails; wrap scalar in list", "TypeError"
    ),
    (
        "def to_percent(val, total):\n    return val / total * 100",
        "def report(counts, total):\n    return {k: to_percent(v, total) for k, v in counts.items()}",
        "def report(counts, total):\n    if total == 0: return {k: 0.0 for k in counts}\n    return {k: to_percent(v, total) for k, v in counts.items()}",
        "Guard zero total before calling to_percent in comprehension", "ZeroDivisionError"
    ),
    # mutation across function boundary
    (
        "def sort_inplace(lst):\n    lst.sort()\n    return lst",
        "def get_sorted_copy(lst):\n    return sort_inplace(lst)",
        "def get_sorted_copy(lst):\n    return sort_inplace(lst.copy())",
        "Must copy before passing to sort_inplace to avoid mutating caller's list", "AssertionError"
    ),
    (
        "def add_item(lst, item):\n    lst.append(item)\n    return lst",
        "def build_list(items, extra):\n    base = []\n    for item in items:\n        result = add_item(base, item)\n    add_item(result, extra)\n    return result",
        "def build_list(items, extra):\n    base = []\n    for item in items:\n        add_item(base, item)\n    add_item(base, extra)\n    return base",
        "result may be unbound if items is empty; always use base directly", "UnboundLocalError"
    ),
    # off-by-one across function call
    (
        "def last_index(lst):\n    return len(lst)",
        "def last_item(lst):\n    idx = last_index(lst)\n    return lst[idx]",
        "def last_item(lst):\n    idx = last_index(lst) - 1\n    return lst[idx]",
        "last_index returns len (one past end); subtract 1 for valid index", "IndexError"
    ),
    (
        "def count(lst):\n    return len(lst) - 1",
        "def avg(lst):\n    return sum(lst) / count(lst)",
        "def avg(lst):\n    n = len(lst)\n    if n == 0: return 0.0\n    return sum(lst) / n",
        "count() had off-by-one; caller should use len() directly with guard", "ZeroDivisionError"
    ),
    # wrong accumulator pattern across calls
    (
        "def add_to(result, value):\n    result.append(value)\n    return result",
        "def collect(items):\n    out = []\n    return [add_to(out, x) for x in items]",
        "def collect(items):\n    out = []\n    for x in items:\n        add_to(out, x)\n    return out",
        "Comprehension collects return values (list of lists); use loop to build out", "AssertionError"
    ),
    (
        "def multiply(a, b):\n    return a * b",
        "def dot_product(v1, v2):\n    return sum(multiply(a, b) for a, b in zip(v1, v2, strict=False))",
        "def dot_product(v1, v2):\n    if len(v1) != len(v2): raise ValueError('lengths must match')\n    return sum(multiply(a, b) for a, b in zip(v1, v2))",
        "Validate equal length vectors before computing dot product", "AssertionError"
    ),
    # caching bug
    (
        "def expensive(n):\n    return n ** 2",
        "cache = {}\ndef cached_expensive(n):\n    if n in cache: return cache[n]\n    result = expensive(n)\n    return result",
        "cache = {}\ndef cached_expensive(n):\n    if n in cache: return cache[n]\n    result = expensive(n)\n    cache[n] = result\n    return result",
        "Result is computed but never stored in cache", "AssertionError"
    ),
    # string processing pipeline
    (
        "def tokenize(text):\n    return text.split()",
        "def word_count(text):\n    tokens = tokenize(text)\n    return len(tokens)",
        "def word_count(text):\n    if not text or not text.strip(): return 0\n    tokens = tokenize(text)\n    return len(tokens)",
        "Guard empty/whitespace-only strings before tokenizing", "AssertionError"
    ),
    (
        "def strip_tags(html):\n    import re\n    return re.sub('<[^>]+>', '', html)",
        "def extract_text(html):\n    text = strip_tags(html)\n    return text.upper()",
        "def extract_text(html):\n    text = strip_tags(html)\n    return text.strip().upper()",
        "Strip whitespace after removing tags to avoid leading/trailing spaces", "AssertionError"
    ),
    # aggregate functions
    (
        "def total(values):\n    return sum(values)",
        "def mean(values):\n    return total(values) / len(values)",
        "def mean(values):\n    if not values: return 0.0\n    return total(values) / len(values)",
        "Guard empty list before dividing; total([]) / len([]) raises ZeroDivisionError", "ZeroDivisionError"
    ),
    (
        "def percentile(data, p):\n    data = sorted(data)\n    idx = int(len(data) * p)\n    return data[idx]",
        "def median(data):\n    return percentile(data, 0.5)",
        "def median(data):\n    if not data: return None\n    data = sorted(data)\n    n = len(data)\n    if n % 2 == 0: return (data[n//2 - 1] + data[n//2]) / 2\n    return data[n//2]",
        "Median of even-length list must average two middle elements", "AssertionError"
    ),
    (
        "def flatten(lst):\n    return [x for sub in lst for x in sub]",
        "def unique_flat(lst):\n    return set(flatten(lst))",
        "def unique_flat(lst):\n    return sorted(set(flatten(lst)))",
        "Return sorted list for deterministic output, not a set", "AssertionError"
    ),
]

# ══════════════════════════════════════════════════════════════════════
# 4. INSTRUCTION TEMPLATES (22 diverse variants)
# ══════════════════════════════════════════════════════════════════════

INSTRUCTION_TEMPLATES = [
    "Expert APR agent. Fix code using stack trace.",
    "You are an automated program repair system. Analyse the error and return only the corrected function.",
    "Repair the following buggy Python function. Output the fixed code only.",
    "Identify and fix the bug in the code below. Use the provided trace as a guide.",
    "You are a senior Python engineer. Fix the logic error and return the corrected function.",
    "Debug the Python function below. Return only the fixed function with no explanation.",
    "Find and correct the single bug in this function. Output the repaired function.",
    "You are a code repair AI. Given a failing test trace, output the fixed Python function.",
    "Fix the bug causing the assertion failure. Return the corrected function body only.",
    "Analyse the traceback and patch the logic error. Output corrected Python code.",
    "You are an expert in Python debugging. Identify the root cause and return the fixed function.",
    "Given the error trace below, repair the function. No explanation — code only.",
    "Your task is automated program repair. Output only the corrected function definition.",
    "A test is failing due to a logic bug. Fix the function and return the corrected version.",
    "You receive a buggy function and its failing test output. Return the corrected function.",
    "Act as a static analysis tool that produces fixes. Output the repaired Python function.",
    "The following function contains a bug revealed by the trace. Fix it and output the function.",
    "Correct the logic error in this Python function. Provide only the fixed code.",
    "You are TestMate, an APR system. Analyse the trace, locate the bug, output the fix.",
    "Given this failing function and traceback, produce the minimal fix. Code only.",
    "Repair this Python function so the test passes. Return only the function definition.",
    "The function below has a bug. Use the stack trace to identify and fix it.",
]

# ══════════════════════════════════════════════════════════════════════
# 5. MULTI-FUNCTION CONTEXT WRAPPERS
# ══════════════════════════════════════════════════════════════════════

HELPER_WRAPPERS = [
    ("class Solution:\n    def helper(self, x):\n        return x * 2\n\n    ",
     "\n\n    def validate(self, x):\n        return x is not None"),
    ("def preprocess(data):\n    return [x for x in data if x is not None]\n\n",
     "\n\ndef postprocess(result):\n    return sorted(result)"),
    ("import math\n\ndef utility(x):\n    return math.floor(x)\n\n", ""),
    ("from typing import List, Optional\n\n", ""),
]

def wrap_in_context(func_source: str) -> str:
    if random.random() < 0.3:
        prefix, suffix = random.choice(HELPER_WRAPPERS)
        if "class " in prefix:
            indented = "\n".join("    " + l for l in func_source.split("\n"))
            return prefix + indented + suffix
        return prefix + func_source + suffix
    return func_source


# ══════════════════════════════════════════════════════════════════════
# 6. DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_function(name: str, params: str, body: str) -> str:
    lines = body.split("\n")
    indented = "\n".join("    " + l for l in lines)
    return f"def {name}({params}):\n{indented}"

def build_and_save_datasets(quixbugs_repeats: int = 30, total_synthetic: int = 8000,
                             total_buginpy: int = 2000, total_multifunc: int = 2000):
    random.seed(42)
    train_data, test_data = [], []

    # 1. QuixBugs train
    for name, data in QUIXBUGS_TRAIN.items():
        for _ in range(quixbugs_repeats):
            buggy_ctx = wrap_in_context(data["buggy"])
            trace = generate_traceback(data["error"], name, data["buggy"], data["issue"])
            train_data.append({
                "instruction": random.choice(INSTRUCTION_TEMPLATES),
                "input":  f"ISSUE: {data['issue']}\n\nTRACE:\n{trace}\n\nCODE:\n{buggy_ctx}",
                "output": data["fixed"],
            })

    # 2. Synthetic patterns
    for _ in range(total_synthetic):
        f_name, params, buggy_body, fixed_body, issue, error_type, _ = random.choice(SYNTHETIC_PATTERNS)
        buggy_fn = build_function(f_name, params, buggy_body)
        fixed_fn = build_function(f_name, params, fixed_body)
        buggy_ctx = wrap_in_context(buggy_fn)
        trace = generate_traceback(error_type, f_name, buggy_fn, issue)
        train_data.append({
            "instruction": random.choice(INSTRUCTION_TEMPLATES),
            "input":  f"ISSUE: {issue}\n\nTRACE:\n{trace}\n\nBUGGY:\n{buggy_ctx}",
            "output": fixed_fn,
        })

    # 3. BugsInPy-inspired patterns
    for _ in range(total_buginpy):
        f_name, params, buggy_body, fixed_body, issue, error_type, _ = random.choice(BUGINPY_PATTERNS)
        buggy_fn = build_function(f_name, params, buggy_body)
        fixed_fn = build_function(f_name, params, fixed_body)
        trace = generate_traceback(error_type, f_name, buggy_fn, issue)
        train_data.append({
            "instruction": random.choice(INSTRUCTION_TEMPLATES),
            "input":  f"ISSUE: {issue}\n\nTRACE:\n{trace}\n\nBUGGY:\n{buggy_fn}",
            "output": fixed_fn,
        })

    # 4. Multi-function bugs
    for _ in range(total_multifunc):
        helper_code, buggy_caller, fixed_caller, issue, error_type = random.choice(MULTI_FUNCTION_BUGS)
        full_buggy = helper_code + "\n\n" + buggy_caller
        trace = generate_traceback(error_type, "caller", buggy_caller, issue)
        train_data.append({
            "instruction": random.choice(INSTRUCTION_TEMPLATES),
            "input":  (
                f"ISSUE: {issue}\n\nTRACE:\n{trace}\n\n"
                f"BUGGY CODE (fix only the second function):\n{full_buggy}"
            ),
            "output": fixed_caller,
        })

    # 5. Held-out QuixBugs test
    for name, data in QUIXBUGS_TEST.items():
        for _ in range(5):
            trace = generate_traceback(data["error"], name, data["buggy"], data["issue"])
            test_data.append({
                "instruction": random.choice(INSTRUCTION_TEMPLATES),
                "input":  f"ISSUE: {data['issue']}\n\nTRACE:\n{trace}\n\nCODE:\n{data['buggy']}",
                "output": data["fixed"],
            })

    random.shuffle(train_data)
    random.shuffle(test_data)

    def save_jsonl(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                prompt = (
                    f"<|im_start|>system\n{item['instruction']}<|im_end|>\n"
                    f"<|im_start|>user\n{item['input']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                )
                f.write(json.dumps({"text": prompt}, ensure_ascii=False) + "\n")

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(test_data,  TEST_FILE)

    print("✅ Dataset built!")
    print(f"   QuixBugs train  : {len(QUIXBUGS_TRAIN)} bugs x {quixbugs_repeats} = {len(QUIXBUGS_TRAIN)*quixbugs_repeats:,}")
    print(f"   Synthetic       : {total_synthetic:,} ({len(SYNTHETIC_PATTERNS)} patterns)")
    print(f"   BugsInPy-style  : {total_buginpy:,} ({len(BUGINPY_PATTERNS)} patterns)")
    print(f"   Multi-function  : {total_multifunc:,} ({len(MULTI_FUNCTION_BUGS)} templates)")
    print(f"   Total train     : {len(train_data):,}")
    print(f"   Test (held-out) : {len(test_data):,} ({len(QUIXBUGS_TEST)} unseen QuixBugs bugs)")

def get_tokenized_dataset(tokenizer):
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": TEST_FILE})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=MAX_LENGTH, padding=False)

    tokenized_ds = dataset.map(tokenize_fn, batched=True,
                               remove_columns=dataset["train"].column_names)
    print(f"✅ Tokenised — train: {len(tokenized_ds['train']):,} | test: {len(tokenized_ds['test']):,}")
    return tokenized_ds

if __name__ == "__main__":
    build_and_save_datasets(
        quixbugs_repeats=30,
        total_synthetic=8000,
        total_buginpy=2000,
        total_multifunc=2000,
    )