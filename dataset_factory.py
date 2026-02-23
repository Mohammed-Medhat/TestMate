import json
import random
import os
from datasets import load_dataset
from config import TRAIN_FILE, TEST_FILE, MAX_LENGTH

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
        "issue": "Fix binary search - loop condition should be lo < hi to avoid index error."
    },
    "gcd": {
        "buggy": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(a % b, b)",
        "fixed": "def gcd(a, b):\n    if b == 0: return a\n    else: return gcd(b, a % b)",
        "issue": "Fix GCD - swap arguments in recursive call: gcd(b, a % b)."
    },
    "flatten": {
        "buggy": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield flatten(x)",
        "fixed": "def flatten(arr):\n    for x in arr:\n        if isinstance(x, list):\n            for y in flatten(x): yield y\n        else: yield x",
        "issue": "Recursion error: yield flatten(x) used on non-list element."
    },
    "reverse_linked_list": {
        "buggy": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        node = nextnode\n    return prevnode",
        "fixed": "def reverse_linked_list(node):\n    prevnode = None\n    while node:\n        nextnode = node.next\n        node.next = prevnode\n        prevnode = node\n        node = nextnode\n    return prevnode",
        "issue": "Pointer error: prevnode must be updated to the current node."
    },
    "lcs": {
        "buggy": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b))",
        "fixed": "def lcs(a, b):\n    if not a or not b: return ''\n    elif a[0] == b[0]: return a[0] + lcs(a[1:], b[1:])\n    else: return max(lcs(a, b[1:]), lcs(a[1:], b), key=len)",
        "issue": "Logic error: max() needs key=len to compare string lengths."
    },
    "bucketsort": {
        "buggy": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(arr):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "fixed": "def bucketsort(arr, k):\n    counts = [0] * k\n    for x in arr: counts[x] += 1\n    sorted_arr = []\n    for i, count in enumerate(counts):\n        sorted_arr.extend([i] * count)\n    return sorted_arr",
        "issue": "Loop error: final iteration should be over 'counts' not 'arr'."
    },
    "hanoi": {
        "buggy": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, helper))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "fixed": "def hanoi(height, start=1, end=3):\n    steps = []\n    if height > 0:\n        helper = ({1, 2, 3} - {start} - {end}).pop()\n        steps.extend(hanoi(height - 1, start, helper))\n        steps.append((start, end))\n        steps.extend(hanoi(height - 1, helper, end))\n    return steps",
        "issue": "Logic error: disc must move to 'end' peg, not 'helper' peg."
    },
    "pascal": {
        "buggy": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "fixed": "def pascal(n):\n    rows = [[1]]\n    for r in range(1, n):\n        row = []\n        for c in range(0, r + 1):\n            v = (rows[r-1][c-1] if c > 0 else 0) + (rows[r-1][c] if c < r else 0)\n            row.append(v)\n        rows.append(row)\n    return rows",
        "issue": "Off-by-one error: column loop must include the last element (r + 1)."
    },
    "rpn_eval": {
        "buggy": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            a = stack.pop(); b = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "fixed": "def rpn_eval(tokens):\n    def op(s, a, b): return {'+':a+b, '-':a-b, '*':a*b, '/':a/b}[s]\n    stack = []\n    for t in tokens:\n        if isinstance(t, float): stack.append(t)\n        else:\n            b = stack.pop(); a = stack.pop()\n            stack.append(op(t, a, b))\n    return stack.pop()",
        "issue": "Operand order: for subtraction/division, the first pop is 'b'."
    },
    "topological_sort": {
        "buggy": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for successor in node.successors:\n            if successor not in ordered_nodes:\n                ordered_nodes.append(successor)\n    return ordered_nodes",
        "fixed": "def topological_sort(nodes):\n    ordered_nodes = []\n    for node in nodes:\n        if not node.predecessors: ordered_nodes.append(node)\n    for node in ordered_nodes:\n        for s in node.successors:\n            if all(p in ordered_nodes for p in s.predecessors) and s not in ordered_nodes:\n                ordered_nodes.append(s)\n    return ordered_nodes",
        "issue": "Logic error: successor must wait until all predecessors are in list."
    },
    "wrap": {
        "buggy": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:]\n    lines.append(text)\n    return lines",
        "fixed": "def wrap(text, cols):\n    lines = []\n    while len(text) > cols:\n        end = text.rfind(' ', 0, cols)\n        if end == -1: end = cols\n        lines.append(text[:end])\n        text = text[end:].strip()\n    lines.append(text)\n    return lines",
        "issue": "Formatting error: missing strip() to remove leading spaces in new lines."
    },
    "next_permutation": {
        "buggy": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] < arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1 :] = reversed(arr[i + 1 :])\n                    return arr\n    return None",
        "fixed": "def next_permutation(arr):\n    for i in range(len(arr) - 2, -1, -1):\n        if arr[i] < arr[i + 1]:\n            for j in range(len(arr) - 1, i, -1):\n                if arr[j] > arr[i]:\n                    arr[i], arr[j] = arr[j], arr[i]\n                    arr[i + 1 :] = reversed(arr[i + 1 :])\n                    return arr\n    return None",
        "issue": "Logic error: swap condition should be arr[j] > arr[i]."
    },
    "kth": {
        "buggy": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr if x < pivot]\n    above = [x for x in arr if x > pivot]\n    num_less = len(below)\n    num_less_or_equal = len(arr) - len(above)\n    if k < num_less: return kth(below, k)\n    elif k >= num_less_or_equal: return kth(above, k - num_less_or_equal)\n    else: return pivot",
        "fixed": "def kth(arr, k):\n    pivot = arr[0]\n    below = [x for x in arr[1:] if x < pivot]\n    above = [x for x in arr[1:] if x >= pivot]\n    num_less = len(below)\n    if k < num_less: return kth(below, k)\n    elif k == num_less: return pivot\n    else: return kth(above, k - num_less - 1)",
        "issue": "Fix kth element selection - pivot must be excluded from recursive slices."
    },
    "shunting_yard": {
        "buggy": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] <= precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "fixed": "def shunting_yard(tokens):\n    precedences = {'+': 1, '-': 1, '*': 2, '/': 2}\n    rpndict = []\n    opstack = []\n    for token in tokens:\n        if isinstance(token, int): rpndict.append(token)\n        else:\n            while opstack and precedences[token] < precedences[opstack[-1]]:\n                rpndict.append(opstack.pop())\n            opstack.append(token)\n    while opstack: rpndict.append(opstack.pop())\n    return rpndict",
        "issue": "Logic: precedence comparison error in operator stacking."
    },
    "subsequences": {
        "buggy": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k):\n        ret.extend([i] + rest for rest in subsequences(i + 1, b, k - 1))\n    return ret",
        "fixed": "def subsequences(a, b, k):\n    if k == 0: return [[]]\n    ret = []\n    for i in range(a, b + 1 - k + 1):\n        ret.extend([[i] + rest for rest in subsequences(i + 1, b, k - 1)])\n    return ret",
        "issue": "Off-by-one: loop range must include b + 1 - k."
    },
    "possible_change": {
        "buggy": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "fixed": "def possible_change(coins, amount):\n    if amount == 0: return 1\n    if amount < 0 or not coins: return 0\n    c, *rest = coins\n    return possible_change(coins, amount - c) + possible_change(rest, amount)",
        "issue": "Logic: Handle negative amount base case in coin change recursion."
    },
    "shortest_path_length": {
        "buggy": "def shortest_path_length(startnode, goalnode):\n    unvisited_nodes = []\n    dist = {startnode: 0}\n    while unvisited_nodes:\n        current_node = min(unvisited_nodes, key=lambda node: dist.get(node, float('inf')))\n        unvisited_nodes.remove(current_node)\n        if current_node is goalnode: return dist[current_node]\n        for nextnode, distance in current_node.successors.items():\n            new_dist = dist[current_node] + distance\n            if new_dist < dist.get(nextnode, float('inf')):\n                dist[nextnode] = new_dist\n    return float('inf')",
        "fixed": "def shortest_path_length(startnode, goalnode):\n    from heapq import heappush, heappop\n    queue = [(0, startnode)]\n    visited = set()\n    while queue:\n        d, node = heappop(queue)\n        if node in visited: continue\n        visited.add(node)\n        if node is goalnode: return d\n        for nextnode, dist in node.successors.items():\n            heappush(queue, (d + dist, nextnode))\n    return float('inf')",
        "issue": "Efficiency error: Dijkstra implementation needs priority queue."
    }
}

def generate_traceback(error_type, func_name, line_no, detail=""):
    trace = f'Traceback (most recent call last):\n'
    trace += f'  File "app/logic.py", line {line_no}, in {func_name}\n'
    if error_type == "ZeroDivisionError":
        trace += f'    return a / b\n{error_type}: division by zero'
    else:
        trace += f'    assert output == expected\n{error_type}: Logic verification failed. {detail}'
    return trace

def build_and_save_datasets(total_synthetic=6000):
    all_data = []
    
    for name, data in QUIXBUGS_TRAIN.items():
        for _ in range(150):
            all_data.append({
                "instruction": "Repair algorithmic logic error.",
                "input": f"ISSUE: {data['issue']}\n\nCODE:\n{data['buggy']}",
                "output": data['fixed']
            })
            
    logic_patterns = [
        ("check_limit", "v, l", "return v > l", "return v >= l", "Off-by-one: inclusive limit required"),
        ("verify_id", "uid", "if uid is 10:", "if uid == 10:", "Identity vs Equality: use == for integers"),
        ("is_active", "data", "if len(data) > 0:", "if data:", "Logic: use implicit boolean for sequences")
    ]
    
    for _ in range(total_synthetic):
        f_name, params, buggy, fixed, issue = random.choice(logic_patterns)
        trace = generate_traceback("AssertionError", f_name, random.randint(10, 50), issue)
        all_data.append({
            "instruction": "Expert APR agent. Fix code using stack trace.",
            "input": f"ISSUE: {issue}\n\nTRACE:\n{trace}\n\nBUGGY:\ndef {f_name}({params}):\n    {buggy}",
            "output": f"def {f_name}({params}):\n    {fixed}"
        })

    random.shuffle(all_data)
    train_data = all_data[:-200]
    test_data = all_data[-200:]

    def save_jsonl(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                prompt = f"<|im_start|>system\n{item['instruction']}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                f.write(json.dumps({"text": prompt}, ensure_ascii=False) + '\n')

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(test_data, TEST_FILE)
    print(f"✅ Production Dataset Ready: {len(all_data)} samples.")

def get_tokenized_dataset(tokenizer):
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "test": TEST_FILE})
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)