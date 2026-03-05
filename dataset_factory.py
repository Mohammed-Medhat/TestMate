import json
import random
import os
from datasets import load_dataset
from config import TRAIN_FILE, TEST_FILE, MAX_LENGTH

# ══════════════════════════════════════════════════════════════════
# QuixBugs — 20 training bugs  (10 reserved for benchmark)
# ══════════════════════════════════════════════════════════════════
QUIXBUGS_TRAIN = {
    "bitcount": {
        "buggy": "def bitcount(n):\n    count = 0\n    while n:\n        n ^= n - 1\n        count += 1\n    return count",
        "fixed": "def bitcount(n):\n    count = 0\n    while n:\n        n &= n - 1\n        count += 1\n    return count",
        "issue": "Fix bitcount - XOR should be AND to correctly remove bits."
    },
    "quicksort": {
        "buggy": "def quicksort(arr):\n    if not arr: return []\n    pivot = arr[0]\n    lesser = quicksort([x for x in arr[1:] if x < pivot])\n    greater = quicksort([x for x in arr[1:] if x > pivot])\n    return lesser + [pivot] + greater",
        "fixed": "def quicksort(arr):\n    if not arr: return []\n    pivot = arr[0]\n    lesser = quicksort([x for x in arr[1:] if x < pivot])\n    greater = quicksort([x for x in arr[1:] if x >= pivot])\n    return lesser + [pivot] + greater",
        "issue": "Fix quicksort - greater partition must handle duplicates using >=."
    },
    "is_valid_parenthesization": {
        "buggy": "def is_valid_parenthesization(parens):\n    depth = 0\n    for p in parens:\n        if p == '(': depth += 1\n        else:\n            depth -= 1\n            if depth < 0: return False\n    return True",
        "fixed": "def is_valid_parenthesization(parens):\n    depth = 0\n    for p in parens:\n        if p == '(': depth += 1\n        else:\n            depth -= 1\n            if depth < 0: return False\n    return depth == 0",
        "issue": "Fix logic - final depth must be 0 for valid parenthesization."
    },
    "find_first_in_sorted": {
        "buggy": "def find_first_in_sorted(arr, x):\n    lo, hi = 0, len(arr)\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if x == arr[mid]: return mid\n        elif x < arr[mid]: hi = mid\n        else: lo = mid + 1\n    return -1",
        "fixed": "def find_first_in_sorted(arr, x):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if x == arr[mid]: return mid\n        elif x < arr[mid]: hi = mid\n        else: lo = mid + 1\n    return -1",
        "issue": "Fix binary search - loop condition should be lo < hi."
    },
    "gcd": {
        "buggy": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(a % b, b)",
        "fixed": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(b, a % b)",
        "issue": "Fix GCD - swap arguments: gcd(b, a % b)."
    },
    "flatten": {
        "buggy": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield flatten(x)",
        "fixed": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield x",
        "issue": "Recursion error: yield flatten(x) used on non-list element."
    },
    "reverse_linked_list": {
        "buggy": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        node = nextnode\n    return prevnode",
        "fixed": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        prevnode = node\n        node = nextnode\n    return prevnode",
        "issue": "Pointer error: prevnode must be updated to current node."
    },
    "lcs": {
        "buggy": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b))",
        "fixed": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b), key=len)",
        "issue": "Logic error: max() needs key=len to compare string lengths."
    },
    "bucketsort": {
        "buggy": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(arr):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "fixed": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(counts):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "issue": "Loop error: iterate over 'counts' not 'arr'."
    },
    "hanoi": {
        "buggy": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, helper))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "fixed": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, end))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "issue": "Logic error: disc must move to 'end' peg not 'helper'."
    },
    "pascal": {
        "buggy": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "fixed": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r + 1):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "issue": "Off-by-one: column loop must go to r + 1."
    },
    "rpn_eval": {
        "buggy": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            a = stack.pop(); b = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "fixed": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            b = stack.pop(); a = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "issue": "Operand order: first pop is 'b', second is 'a'."
    },
    "topological_sort": {
        "buggy": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for successor in node.successors:\n            if successor not in ordered_nodes:\n                ordered_nodes.append(successor)\n    return ordered_nodes",
        "fixed": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for s in node.successors:\n            if all(p in ordered_nodes for p in s.predecessors) and s not in ordered_nodes:\n                ordered_nodes.append(s)\n    return ordered_nodes",
        "issue": "Logic: successor must wait for all predecessors."
    },
    "wrap": {
        "buggy": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:]\n    lines.append(text)\n    return lines",
        "fixed": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:].strip()\n    lines.append(text)\n    return lines",
        "issue": "Missing strip() to remove leading spaces."
    },
    "next_permutation": {
        "buggy": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] < arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1 :] = reversed(arr[i + 1 :])\n                    return arr\n    return None",
        "fixed": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] > arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1 :] = reversed(arr[i + 1 :])\n                    return arr\n    return None",
        "issue": "Swap condition should be arr[j] > arr[i]."
    },
    "kth": {
        "buggy": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr if x < pivot]\n    above = [x for x in arr if x > pivot]\n    num_less = len(below)\n    num_less_or_equal = len(arr) - len(above)\n    if k < num_less: return kth(below, k)\n    elif k >= num_less_or_equal: return kth(above, k - num_less_or_equal)\n    else: return pivot",
        "fixed": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr[1:] if x < pivot]\n    above = [x for x in arr[1:] if x >= pivot]\n    num_less = len(below)\n    if k < num_less: return kth(below, k)\n    elif k == num_less: return pivot\n    else: return kth(above, k - num_less - 1)",
        "issue": "Pivot must be excluded from recursive slices."
    },
    "shunting_yard": {
        "buggy": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] <= precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "fixed": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] < precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "issue": "Precedence comparison should be < not <=."
    },
    "subsequences": {
        "buggy": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k):\n        ret.extend([i] + rest for rest in subsequences(i + 1, b, k - 1))\n    return ret",
        "fixed": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k + 1):\n        ret.extend([[i] + rest for rest in subsequences(i + 1, b, k - 1)])\n    return ret",
        "issue": "Off-by-one: loop range must include b + 1 - k."
    },
    "possible_change": {
        "buggy": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "fixed": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if amount < 0 or not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "issue": "Missing base case: handle negative amount."
    },
    "shortest_path_length": {
        "buggy": "def shortest_path_length(startnode, goalnode):\n    unvisited_nodes = []\n    dist = {startnode: 0}\n    while unvisited_nodes:\n        current_node = min(unvisited_nodes, key=lambda node: dist.get(node, float('inf')))\n        unvisited_nodes.remove(current_node)\n        if current_node is goalnode: return dist[current_node]\n        for nextnode, distance in current_node.successors.items():\n            new_dist = dist[current_node] + distance\n            if new_dist < dist.get(nextnode, float('inf')):\n                dist[nextnode] = new_dist\n    return float('inf')",
        "fixed": "def shortest_path_length(startnode, goalnode):\n    from heapq import heappush, heappop\n    queue = [(0, startnode)]\n    visited = set()\n    while queue:\n        d, node = heappop(queue)\n        if node in visited: continue\n        visited.add(node)\n        if node is goalnode: return d\n        for nextnode, dist in node.successors.items():\n            heappush(queue, (d + dist, nextnode))\n    return float('inf')",
        "issue": "Dijkstra needs a priority queue."
    },
}

# ══════════════════════════════════════════════════════════════════
# 63 Synthetic Bug Patterns across 9 categories
# (func_name, params, buggy_body, fixed_body, issue, error_type, category)
# ══════════════════════════════════════════════════════════════════
SYNTHETIC_PATTERNS = [
    # Off-by-one
    ("check_limit",   "v, l",    "return v > l",                         "return v >= l",                           "Off-by-one: inclusive upper limit requires >=",            "AssertionError",    "off_by_one"),
    ("check_lower",   "v, l",    "return v < l",                         "return v <= l",                           "Off-by-one: inclusive lower limit requires <=",            "AssertionError",    "off_by_one"),
    ("index_last",    "arr",     "return arr[len(arr)]",                  "return arr[len(arr) - 1]",                "Index error: last element is at len-1",                   "IndexError",        "off_by_one"),
    ("slice_head",    "arr, k",  "return arr[:k-1]",                     "return arr[:k]",                          "Slice error: arr[:k] correctly includes index k-1",       "AssertionError",    "off_by_one"),
    ("count_items",   "arr",     "return len(arr) - 1",                  "return len(arr)",                         "Off-by-one: length must not be decremented",              "AssertionError",    "off_by_one"),
    ("mid_index",     "lo, hi",  "return (lo + hi) / 2",                 "return (lo + hi) // 2",                   "Type error: integer division needed for index",           "TypeError",         "off_by_one"),
    ("loop_n",        "n",       "return list(range(1, n))",             "return list(range(n))",                   "Off-by-one: range(n) gives n items, range(1,n) gives n-1","AssertionError",    "off_by_one"),
    # Logic
    ("verify_id",     "uid",     "return uid is 10",                     "return uid == 10",                        "Identity vs Equality: use == not 'is' for integers",      "AssertionError",    "logic"),
    ("is_even",       "n",       "return n % 2 == 1",                    "return n % 2 == 0",                       "Logic error: even numbers have remainder 0",               "AssertionError",    "logic"),
    ("is_positive",   "n",       "return n > 1",                         "return n > 0",                            "Logic error: positive includes 1, use > 0",               "AssertionError",    "logic"),
    ("all_true",      "flags",   "return any(flags)",                    "return all(flags)",                       "Logic: all() requires every flag to be True",             "AssertionError",    "logic"),
    ("in_range",      "x, a, b", "return a < x < b",                    "return a <= x <= b",                      "Logic: inclusive range check requires <=",                "AssertionError",    "logic"),
    ("toggle",        "flag",    "return flag",                          "return not flag",                         "Logic error: toggle must negate the boolean",             "AssertionError",    "logic"),
    ("safe_div",      "a, b",    "return a / b",                         "return a / b if b != 0 else 0",           "ZeroDivisionError: guard against zero denominator",       "ZeroDivisionError", "logic"),
    ("negate",        "n",       "return n * -0",                        "return n * -1",                           "Logic error: -0 is 0, use -1 to negate",                  "AssertionError",    "logic"),
    ("abs_val",       "n",       "return n if n > 0 else n",             "return n if n >= 0 else -n",              "Logic: absolute value must negate negative numbers",      "AssertionError",    "logic"),
    ("clamp",         "x, lo, hi","return max(lo, x)",                   "return max(lo, min(x, hi))",              "Logic: clamp must also apply upper bound",                "AssertionError",    "logic"),
    ("xor_check",     "a, b",    "return a and b",                       "return bool(a) ^ bool(b)",                "Logic: XOR differs from AND",                             "AssertionError",    "logic"),
    # Collection
    ("is_active",     "data",    "return len(data) > 0",                 "return bool(data)",                       "Logic: use implicit boolean for non-empty sequences",     "AssertionError",    "collection"),
    ("first_item",    "arr",     "return arr[1]",                        "return arr[0]",                           "Index error: first item is at index 0",                   "IndexError",        "collection"),
    ("safe_get",      "d, k",    "return d[k]",                          "return d.get(k)",                         "KeyError: use dict.get() for safe access",                "KeyError",          "collection"),
    ("append_extend", "lst, items","lst.append(items)\n    return lst",  "lst.extend(items)\n    return lst",       "Logic: append adds one object, extend adds elements",     "AssertionError",    "collection"),
    ("remove_dup",    "arr",     "return list(arr)",                     "return list(set(arr))",                   "Logic: deduplication requires converting to set",         "AssertionError",    "collection"),
    ("dict_assign",   "d, k, v", "d[k] == v\n    return d",             "d[k] = v\n    return d",                  "Assignment error: == compares, = assigns",                "AssertionError",    "collection"),
    ("stack_top",     "stack",   "return stack[0]",                      "return stack[-1]",                        "Stack: LIFO means last element is at index -1",           "AssertionError",    "collection"),
    ("count_key",     "d, k",    "return d[k]",                          "return d.get(k, 0)",                      "KeyError: missing key should default to 0",               "KeyError",          "collection"),
    ("flatten_one",   "lst",     "return [x for x in lst]",             "return [x for sub in lst for x in sub]",  "Logic: nested list needs double iteration",               "AssertionError",    "collection"),
    # String
    ("is_empty_str",  "s",       "return s == None",                     "return s == ''",                          "Type error: empty string is '' not None",                 "AssertionError",    "string"),
    ("trim",          "s",       "return s.strip",                       "return s.strip()",                        "Call error: strip is a method, must use ()",              "TypeError",         "string"),
    ("starts_with",   "s, prefix","return prefix in s",                  "return s.startswith(prefix)",             "Logic: 'in' checks anywhere, startswith checks start",   "AssertionError",    "string"),
    ("reverse_str",   "s",       "return s.reverse()",                   "return s[::-1]",                          "AttributeError: strings have no .reverse()",             "AttributeError",    "string"),
    ("char_at",       "s, i",    "return s[i:]",                         "return s[i]",                             "Slice vs index: s[i] gets character",                     "AssertionError",    "string"),
    ("upper_call",    "s",       "return s.upper",                       "return s.upper()",                        "Call error: upper() needs parentheses",                   "TypeError",         "string"),
    ("repeat_str",    "s, n",    "return s * 0",                         "return s * n",                            "Logic: multiplier must be n not 0",                       "AssertionError",    "string"),
    ("str_contains",  "s, sub",  "return s == sub",                      "return sub in s",                         "Logic: == checks equality, 'in' checks containment",     "AssertionError",    "string"),
    # Recursion
    ("factorial",     "n",       "if n == 0: return 0\n    return n * factorial(n - 1)",   "if n == 0: return 1\n    return n * factorial(n - 1)",   "Base case: factorial(0) must return 1",    "AssertionError", "recursion"),
    ("fib",           "n",       "if n <= 1: return 0\n    return fib(n-1) + fib(n-2)",   "if n <= 1: return n\n    return fib(n-1) + fib(n-2)",   "Base case: fib(1) must return 1",          "AssertionError", "recursion"),
    ("power",         "base, exp","if exp == 0: return 0\n    return base * power(base, exp-1)", "if exp == 0: return 1\n    return base * power(base, exp-1)", "Base case: base^0 is 1 not 0",    "AssertionError", "recursion"),
    ("depth",         "node",    "if not node: return 1\n    return 1 + max(depth(node.left), depth(node.right))", "if not node: return 0\n    return 1 + max(depth(node.left), depth(node.right))", "Base case: depth of empty node is 0", "AssertionError","recursion"),
    ("count_nodes",   "node",    "if not node: return 1\n    return 1 + count_nodes(node.left) + count_nodes(node.right)", "if not node: return 0\n    return 1 + count_nodes(node.left) + count_nodes(node.right)", "Base case: empty node count is 0", "AssertionError","recursion"),
    # Type
    ("parse_int",     "s",       "return float(s)",                      "return int(s)",                           "Type error: integer parsing needs int() not float()",     "AssertionError",    "type"),
    ("to_bool",       "val",     "return val == True",                   "return bool(val)",                        "Type: bool() handles truthy values",                      "AssertionError",    "type"),
    ("none_check",    "val",     "return val == None",                   "return val is None",                      "Use 'is None' not '== None'",                             "AssertionError",    "type"),
    ("int_div",       "a, b",    "return a / b",                         "return a // b",                           "Type: integer division requires //",                      "AssertionError",    "type"),
    ("cast_round",    "x",       "return int(x)",                        "return round(x)",                         "Logic: round() for nearest integer",                      "AssertionError",    "type"),
    ("to_str",        "n",       "return n",                             "return str(n)",                           "Type: must convert number to string",                     "TypeError",         "type"),
    # Loop
    ("find_max",      "arr",     "best = 0\n    for x in arr:\n        if x > best: best = x\n    return best",   "best = arr[0]\n    for x in arr:\n        if x > best: best = x\n    return best",  "Init: use arr[0] not 0",    "AssertionError","loop"),
    ("product",       "arr",     "result = 0\n    for x in arr:\n        result *= x\n    return result",         "result = 1\n    for x in arr:\n        result *= x\n    return result",             "Init: product identity is 1","AssertionError","loop"),
    ("collect_evens", "arr",     "return [x for x in arr if x % 2 == 1]",               "return [x for x in arr if x % 2 == 0]",               "Filter: even numbers have remainder 0",    "AssertionError","loop"),
    ("running_total", "arr",     "out = []\n    s = 0\n    for x in arr:\n        out.append(x)\n        s += x\n    return out",  "out = []\n    s = 0\n    for x in arr:\n        s += x\n        out.append(s)\n    return out",  "Order: accumulate before appending","AssertionError","loop"),
    ("sum_squares",   "n",       "total = 0\n    for i in range(n):\n        total += i\n    return total",       "total = 0\n    for i in range(n):\n        total += i * i\n    return total",       "Logic: accumulate i*i not i",  "AssertionError","loop"),
    ("reverse_list",  "arr",     "return arr.sort()",                    "return arr[::-1]",                        "sort() sorts in-place and returns None",                  "AssertionError",    "loop"),
    # Guard
    ("safe_len",      "s",       "return len(s)",                        "return len(s) if s else 0",               "Guard: handle None/empty before len()",                   "TypeError",         "guard"),
    ("first_or_none", "arr",     "return arr[0]",                        "return arr[0] if arr else None",          "Guard: return None for empty list",                       "IndexError",        "guard"),
    ("max_or_zero",   "arr",     "return max(arr)",                      "return max(arr) if arr else 0",           "Guard: max() raises on empty sequence",                   "ValueError",        "guard"),
    ("safe_sqrt",     "x",       "import math\n    return math.sqrt(x)", "import math\n    return math.sqrt(x) if x >= 0 else 0", "Guard: sqrt undefined for negatives",    "ValueError",   "guard"),
    # Scope / mutation
    ("copy_list",     "lst",     "new = lst\n    return new",            "new = lst.copy()\n    return new",        "Reference: assignment copies reference not values",       "AssertionError",    "scope"),
    ("swap_vals",     "a, b",    "a = b\n    b = a\n    return a, b",    "a, b = b, a\n    return a, b",           "Swap: use tuple unpacking",                               "AssertionError",    "scope"),
    ("clear_list",    "lst",     "lst = []\n    return lst",             "lst.clear()\n    return lst",             "Scope: rebinding local name doesn't clear original",      "AssertionError",    "scope"),
]

# ══════════════════════════════════════════════════════════════════
# Instruction templates
# ══════════════════════════════════════════════════════════════════
INSTRUCTION_TEMPLATES = [
    "Expert APR agent. Fix code using stack trace.",
    "You are an automated program repair system. Analyse the error and return only the corrected function.",
    "Repair the following buggy Python function. Output the fixed code only.",
    "Identify and fix the bug in the code below. Use the provided trace as a guide.",
    "You are a senior Python engineer. Fix the logic error and return the corrected function.",
]


def generate_traceback(error_type, func_name, line_no, detail=""):
    trace = "Traceback (most recent call last):\n"
    trace += f'  File "app/logic.py", line {line_no}, in {func_name}\n'
    msgs = {
        "ZeroDivisionError": "    return a / b\nZeroDivisionError: division by zero",
        "IndexError":        "    return arr[idx]\nIndexError: list index out of range",
        "KeyError":          "    return d[k]\nKeyError: key not found",
        "AttributeError":    "    return obj.method()\nAttributeError: object has no such attribute",
        "ValueError":        "    return math.sqrt(x)\nValueError: math domain error",
        "TypeError":         f"    result = func(arg)\nTypeError: {detail}",
    }
    trace += msgs.get(error_type, f"    assert output == expected\nAssertionError: Logic verification failed. {detail}")
    return trace


def build_and_save_datasets(quixbugs_repeats=200, total_synthetic=8000):
    quixbugs_data, synthetic_data = [], []

    for name, data in QUIXBUGS_TRAIN.items():
        for _ in range(quixbugs_repeats):
            quixbugs_data.append({
                "instruction": random.choice(INSTRUCTION_TEMPLATES),
                "input":  f"ISSUE: {data['issue']}\n\nCODE:\n{data['buggy']}",
                "output": data['fixed']
            })

    for _ in range(total_synthetic):
        f_name, params, buggy_body, fixed_body, issue, error_type, _ = random.choice(SYNTHETIC_PATTERNS)
        line_no    = random.randint(5, 60)
        trace      = generate_traceback(error_type, f_name, line_no, issue)
        buggy_code = f"def {f_name}({params}):\n" + "\n".join(f"    {l}" for l in buggy_body.split("\n"))
        fixed_code = f"def {f_name}({params}):\n" + "\n".join(f"    {l}" for l in fixed_body.split("\n"))
        synthetic_data.append({
            "instruction": random.choice(INSTRUCTION_TEMPLATES),
            "input":  f"ISSUE: {issue}\n\nTRACE:\n{trace}\n\nBUGGY:\n{buggy_code}",
            "output": fixed_code
        })

    random.shuffle(synthetic_data)
    test_data  = synthetic_data[-300:]         # test = synthetic only (no QuixBug leakage)
    train_data = quixbugs_data + synthetic_data[:-300]
    random.shuffle(train_data)

    def save_jsonl(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                prompt = (
                    f"<|im_start|>system\n{item['instruction']}<|im_end|>\n"
                    f"<|im_start|>user\n{item['input']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                )
                f.write(json.dumps({"text": prompt}, ensure_ascii=False) + '\n')

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(test_data,  TEST_FILE)
    print(f"✅ Dataset ready!")
    print(f"   QuixBugs : {len(quixbugs_data):,} samples ({quixbugs_repeats}x augmentation)")
    print(f"   Synthetic: {total_synthetic:,} samples ({len(SYNTHETIC_PATTERNS)} patterns)")
    print(f"   Train    : {len(train_data):,}")
    print(f"   Test     : {len(test_data):,}")


def get_tokenized_dataset(tokenizer):
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": TEST_FILE})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    print(f'✅ Dataset tokenized — train: {len(tokenized_ds["train"]):,} | test: {len(tokenized_ds["test"]):,}')
    return tokenized_ds