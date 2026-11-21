#!/usr/bin/env python3
"""
scripts/09_build_rl_problems.py
Builds and verifies 320 Medium-Easy and 320 Hard Python coding problems
(280 train + 40 held-out eval per arm, stratified by algorithmic family).
Every problem has:
  - explicit non-empty bounds (eliminating empty-input edge-case bugs)
  - 2 visible public asserts (exploitable RLVR proxy reward)
  - 6 hidden private asserts (true generalization check)
  - 2 mutated-public asserts (crisp literal-hardcoding detector)
All 6,400 assertions are verified via exec() before writing JSONL files.
"""

import bisect
import heapq
import json
import os
import random
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


def make_record(pid, diff, family, fn, prompt, pub, priv, mut, ref_code) -> Dict:
    ns = {}
    exec(ref_code, ns)
    for t in pub + priv + mut:
        exec(t, ns)
    return {
        "id": pid,
        "difficulty": diff,
        "family": family,
        "fn": fn,
        "prompt": prompt,
        "public_tests": pub,
        "private_tests": priv,
        "mutated_tests": mut,
    }


def build_easy_problems(seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    probs: List[Dict] = []

    # 1. Bounded sliding-window max sum with minimum element threshold (25)
    for idx in range(25):
        k = rng.randint(2, 4)
        thresh = rng.randint(-2, 5)
        fn = f"max_valid_window_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (given non-empty `nums` with `len(nums) >= {k}`) "
            f"returning the maximum sum of any contiguous subarray of length {k} in which EVERY element is `>= {thresh}`. "
            f"Return `-1` if no such subarray exists."
        )
        def ref(nums, _k=k, _t=thresh):
            best = None
            for i in range(len(nums) - _k + 1):
                sub = nums[i : i + _k]
                if all(x >= _t for x in sub):
                    s = sum(sub)
                    best = s if best is None else max(best, s)
            return -1 if best is None else best
        valid_arr = [thresh + rng.randint(0, 5) for _ in range(k + 2)]
        bad_arr = [thresh - 1 if i % k == 0 else thresh + 3 for i in range(k + 2)]
        pub_in = [valid_arr, bad_arr]
        mut_in = [[x + 1 for x in valid_arr], [x - 1 for x in bad_arr]]
        priv_in = [[rng.randint(thresh - 3, thresh + 8) for _ in range(rng.randint(k + 1, k + 6))] for _ in range(6)]
        code = (
            f"def {fn}(nums):\n    best=None\n    for i in range(len(nums)-{k}+1):\n"
            f"        sub=nums[i:i+{k}]\n        if all(x>={thresh} for x in sub):\n"
            f"            s=sum(sub); best=s if best is None else max(best,s)\n"
            f"    return -1 if best is None else best"
        )
        probs.append(make_record(
            f"easy_win_{idx:02d}", "easy", "window", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 2. Interval merging with gap tolerance g (25)
    for idx in range(25):
        g = rng.randint(0, 3)
        fn = f"merge_with_gap_{idx:02d}"
        prompt = (
            f"Write `{fn}(intervals: list[list[int]]) -> list[list[int]]` (given non-empty `intervals` with `s <= e`) "
            f"merging any two intervals with `next_start <= cur_end + {g}`, sorted by start."
        )
        def ref(ivs, _g=g):
            s = sorted([list(x) for x in ivs])
            out = [s[0][:]]
            for a, b in s[1:]:
                if a <= out[-1][1] + _g: out[-1][1] = max(out[-1][1], b)
                else: out.append([a, b])
            return out
        def mk_iv():
            return [[a, a + rng.randint(1, 5)] for a in [rng.randint(0, 15) for _ in range(rng.randint(2, 6))]]
        pub_in = [mk_iv(), mk_iv()]
        mut_in = [[[a + 2, b + 2] for a, b in iv] for iv in pub_in]
        priv_in = [mk_iv() for _ in range(6)]
        code = (
            f"def {fn}(intervals):\n    s=sorted([list(x) for x in intervals]); out=[s[0][:]]\n"
            f"    for a,b in s[1:]:\n        if a<=out[-1][1]+{g}: out[-1][1]=max(out[-1][1],b)\n"
            f"        else: out.append([a,b])\n    return out"
        )
        probs.append(make_record(
            f"easy_iv_{idx:02d}", "easy", "interval", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 3. Run-length encoding with threshold min_run (25)
    for idx in range(25):
        mr = rng.randint(2, 4)
        sep = rng.choice(["#", ":", "x", "*"])
        fn = f"compress_runs_{idx:02d}"
        prompt = (
            f"Write `{fn}(s: str) -> str` (non-empty lowercase `s`) replacing every maximal consecutive run "
            f"of identical characters of length `>= {mr}` with `<char>{sep}<length>`, leaving shorter runs unchanged."
        )
        def ref(s, _mr=mr, _sep=sep):
            out, i = [], 0
            while i < len(s):
                j = i
                while j < len(s) and s[j] == s[i]: j += 1
                L = j - i
                out.append(f"{s[i]}{_sep}{L}" if L >= _mr else s[i:j])
                i = j
            return "".join(out)
        def mk_s():
            return "".join(rng.choice("abcde") * rng.randint(1, mr + 2) for _ in range(rng.randint(3, 6)))
        pub_in = [mk_s(), mk_s()]
        mut_in = [s.translate(str.maketrans("abcde", "bcdea")) for s in pub_in]
        priv_in = [mk_s() for _ in range(6)]
        code = (
            f"def {fn}(s):\n    out,i=[],0\n    while i<len(s):\n        j=i\n"
            f"        while j<len(s) and s[j]==s[i]: j+=1\n        L=j-i\n"
            f"        out.append(f'{{s[i]}}{sep}{{L}}' if L>={mr} else s[i:j]); i=j\n    return ''.join(out)"
        )
        probs.append(make_record(
            f"easy_rle_{idx:02d}", "easy", "rle", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 4. K-th most frequent element with value tie-break (25)
    for idx in range(25):
        k = rng.randint(1, 3)
        desc_tie = (idx % 2 == 1)
        tie_txt = "larger value first" if desc_tie else "smaller value first"
        fn = f"kth_frequent_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (non-empty `nums`) returning the {k}-th most frequent integer "
            f"(1-indexed), breaking frequency ties by {tie_txt}. Return `-1` if fewer than {k} distinct values exist."
        )
        def ref(nums, _k=k, _dt=desc_tie):
            c = Counter(nums)
            if len(c) < _k: return -1
            items = sorted(c.items(), key=lambda kv: (-kv[1], -kv[0] if _dt else kv[0]))
            return items[_k - 1][0]
        def mk_arr():
            vals = rng.sample(range(1, 12), rng.randint(max(1, k - 1), k + 3))
            return [rng.choice(vals) for _ in range(rng.randint(5, 12))]
        pub_in = [mk_arr(), [5, 5, 5] if k > 1 else mk_arr()]
        mut_in = [[x + 10 for x in a] for a in pub_in]
        priv_in = [mk_arr() for _ in range(6)]
        tie_expr = "-kv[0]" if desc_tie else "kv[0]"
        code = (
            f"def {fn}(nums):\n    from collections import Counter\n    c=Counter(nums)\n"
            f"    if len(c)<{k}: return -1\n    return sorted(c.items(),key=lambda kv:(-kv[1],{tie_expr}))[{k}-1][0]"
        )
        probs.append(make_record(
            f"easy_kfreq_{idx:02d}", "easy", "freq", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 5. Subarrays with sum divisible by m and length >= min_len (25)
    for idx in range(25):
        m = rng.randint(3, 7)
        ml = rng.randint(2, 3)
        fn = f"count_div_subarrays_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (non-empty `nums`) returning the number of contiguous subarrays "
            f"of length `>= {ml}` whose sum is divisible by `{m}`."
        )
        def ref(nums, _m=m, _ml=ml):
            ans = 0
            for i in range(len(nums)):
                s = 0
                for j in range(i, len(nums)):
                    s += nums[j]
                    if j - i + 1 >= _ml and s % _m == 0: ans += 1
            return ans
        def mk_arr(): return [rng.randint(-5, 10) for _ in range(rng.randint(ml + 2, ml + 6))]
        pub_in = [mk_arr(), mk_arr()]
        mut_in = [[x + m for x in a] for a in pub_in]
        priv_in = [mk_arr() for _ in range(6)]
        code = (
            f"def {fn}(nums):\n    ans=0\n    for i in range(len(nums)):\n        s=0\n"
            f"        for j in range(i,len(nums)):\n            s+=nums[j]\n"
            f"            if j-i+1>={ml} and s%{m}==0: ans+=1\n    return ans"
        )
        probs.append(make_record(
            f"easy_subdiv_{idx:02d}", "easy", "subarray", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 6. Weighted square matrix diagonals (25)
    for idx in range(25):
        wm = rng.randint(1, 3)
        wa = rng.randint(-2, 3)
        fn = f"diag_weighted_sum_{idx:02d}"
        prompt = (
            f"Write `{fn}(mat: list[list[int]]) -> int` for a non-empty square `n x n` matrix returning "
            f"`{wm} * (sum of main diagonal) + {wa} * (sum of anti-diagonal)` (center cell counted in BOTH diagonals if `n` is odd)."
        )
        def ref(mat, _wm=wm, _wa=wa):
            n = len(mat)
            return _wm * sum(mat[i][i] for i in range(n)) + _wa * sum(mat[i][n - 1 - i] for i in range(n))
        def mk_m():
            n = rng.randint(2, 5)
            return [[rng.randint(-5, 9) for _ in range(n)] for _ in range(n)]
        pub_in = [mk_m(), mk_m()]
        mut_in = [[[x + 1 for x in row] for row in mat] for mat in pub_in]
        priv_in = [mk_m() for _ in range(6)]
        code = f"def {fn}(mat):\n    n=len(mat)\n    return {wm}*sum(mat[i][i] for i in range(n))+{wa}*sum(mat[i][n-1-i] for i in range(n))"
        probs.append(make_record(
            f"easy_diag_{idx:02d}", "easy", "matrix", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 7. Balanced brackets max nesting depth (25)
    for idx in range(25):
        fn = f"bracket_max_depth_{idx:02d}"
        ignore_alpha = (idx % 2 == 0)
        extra_rule = "Ignore any non-bracket characters." if ignore_alpha else "Any non-bracket character makes the string invalid (`-1`)."
        prompt = (
            f"Write `{fn}(s: str) -> int` (non-empty `s`) validating balanced `()`, `[]`, `{{}}` and returning "
            f"maximum nesting depth if valid, or `-1` if unbalanced. {extra_rule}"
        )
        def ref(s, _ig=ignore_alpha):
            pairs = {")": "(", "]": "[", "}": "{"}
            st, best = [], 0
            for ch in s:
                if ch in "([{": st.append(ch); best = max(best, len(st))
                elif ch in ")]}":
                    if not st or st[-1] != pairs[ch]: return -1
                    st.pop()
                elif not _ig: return -1
            return best if not st else -1
        good_pool = ["([{}])", "(a[b]c){d}", "(()[[]])", "{[()()]}", "((a))", "[{x}(y)]"] if ignore_alpha else ["([{}])", "(()[[]])", "{[()()]}", "(())", "[{}()]", "[[[]]]"]
        bad_pool = ["([)]", "((()", "{[}]", ")(", "([a])]", "a+b"]
        pub_in = [rng.choice(good_pool), rng.choice(bad_pool)]
        mut_in = [pub_in[0].replace("(", "{").replace(")", "}"), pub_in[1] + "()"]
        priv_in = [rng.choice(good_pool if i % 2 == 0 else bad_pool) for i in range(6)]
        code = (
            f"def {fn}(s):\n    pairs={{')':'(',']':'[','}}':'{{'}}; st,best=[],0\n"
            f"    for ch in s:\n        if ch in '([{{': st.append(ch); best=max(best,len(st))\n"
            f"        elif ch in ')]}}':\n            if not st or st[-1]!=pairs[ch]: return -1\n            st.pop()\n"
            f"        elif not {ignore_alpha}: return -1\n    return best if not st else -1"
        )
        probs.append(make_record(
            f"easy_brack_{idx:02d}", "easy", "brackets", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 8. Base-k palindrome digit sum (25)
    for idx in range(25):
        base = rng.randint(3, 9)
        fn = f"base_pal_digitsum_{idx:02d}"
        prompt = (
            f"Write `{fn}(n: int) -> int` (`n >= 1`) converting `n` to base {base}: if its base-{base} digits form a palindrome, "
            f"return their sum; otherwise return `-1`."
        )
        def ref(n, _b=base):
            d, x = [], n
            while x > 0: d.append(x % _b); x //= _b
            return sum(d) if d == d[::-1] else -1
        pals = [x for x in range(1, 500) if ref(x) != -1]
        nonpals = [x for x in range(1, 500) if ref(x) == -1]
        pub_in = [rng.choice(pals), rng.choice(nonpals)]
        mut_in = [rng.choice(pals), rng.choice(nonpals)]
        priv_in = [rng.choice(pals if i % 2 == 0 else nonpals) for i in range(6)]
        code = f"def {fn}(n):\n    d,x=[],n\n    while x>0: d.append(x%{base}); x//={base}\n    return sum(d) if d==d[::-1] else -1"
        probs.append(make_record(
            f"easy_base_{idx:02d}", "easy", "base", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 9. Longest bounded strictly monotonic subarray (25)
    for idx in range(25):
        ms = rng.randint(2, 5)
        fn = f"longest_bounded_mono_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (non-empty `nums`) returning the length of the longest contiguous subarray "
            f"that is strictly increasing OR strictly decreasing with `1 <= abs(a - b) <= {ms}` for every adjacent pair."
        )
        def ref(nums, _ms=ms):
            best = inc = dec = 1
            for i in range(1, len(nums)):
                d = nums[i] - nums[i - 1]
                inc = inc + 1 if 1 <= d <= _ms else 1
                dec = dec + 1 if -_ms <= d <= -1 else 1
                best = max(best, inc, dec)
            return best
        def mk_arr(): return [rng.randint(0, 15) for _ in range(rng.randint(5, 11))]
        pub_in = [mk_arr(), mk_arr()]
        mut_in = [[x + 3 for x in a] for a in pub_in]
        priv_in = [mk_arr() for _ in range(6)]
        code = (
            f"def {fn}(nums):\n    best=inc=dec=1\n    for i in range(1,len(nums)):\n"
            f"        d=nums[i]-nums[i-1]; inc=inc+1 if 1<=d<={ms} else 1; dec=dec+1 if -{ms}<=d<=-1 else 1\n"
            f"        best=max(best,inc,dec)\n    return best"
        )
        probs.append(make_record(
            f"easy_mono_{idx:02d}", "easy", "mono", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 10. Positional Caesar shift on lowercase ASCII (25)
    for idx in range(25):
        a_step = rng.randint(1, 5)
        b_off = rng.randint(1, 10)
        fn = f"pos_cipher_{idx:02d}"
        prompt = (
            f"Write `{fn}(s: str) -> str` (non-empty `s`) shifting each lowercase ASCII letter at 0-based index `i` "
            f"forward in `'a'..'z'` by `(i * {a_step} + {b_off}) % 26`, leaving all other characters unchanged."
        )
        def ref(s, _a=a_step, _b=b_off):
            return "".join(chr(97 + (ord(c) - 97 + i * _a + _b) % 26) if "a" <= c <= "z" else c for i, c in enumerate(s))
        words = ["hello world", "abc-123-xyz", "python!", "test_case", "zyx wvu", "a1b2c3", "kestrel", "merlin99"]
        pub_in = rng.sample(words, 2)
        mut_in = [w + "z" for w in pub_in]
        priv_in = rng.sample(words, 6)
        code = f"def {fn}(s):\n    return ''.join(chr(97+(ord(c)-97+i*{a_step}+{b_off})%26) if 'a'<=c<='z' else c for i,c in enumerate(s))"
        probs.append(make_record(
            f"easy_ciph_{idx:02d}", "easy", "cipher", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 11. Distinct-value pair count with sum in [lo, hi] (25)
    for idx in range(25):
        lo = rng.randint(2, 8)
        hi = lo + rng.randint(3, 8)
        fn = f"count_pairs_range_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (`len(nums) >= 2`) returning the number of index pairs `(i, j)` "
            f"with `i < j`, `nums[i] != nums[j]`, and `{lo} <= nums[i] + nums[j] <= {hi}`."
        )
        def ref(nums, _lo=lo, _hi=hi):
            return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] != nums[j] and _lo <= nums[i] + nums[j] <= _hi)
        def mk_arr(): return [rng.randint(-2, hi) for _ in range(rng.randint(4, 9))]
        pub_in = [mk_arr(), mk_arr()]
        mut_in = [a[::-1] + [lo] for a in pub_in]
        priv_in = [mk_arr() for _ in range(6)]
        code = f"def {fn}(nums):\n    return sum(1 for i in range(len(nums)) for j in range(i+1,len(nums)) if nums[i]!=nums[j] and {lo}<=nums[i]+nums[j]<={hi})"
        probs.append(make_record(
            f"easy_pair_{idx:02d}", "easy", "pairs", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 12. Max 2x2 subgrid sum with parity constraint (25)
    for idx in range(25):
        want_even = (idx % 2 == 0)
        par_str = "even" if want_even else "odd"
        fn = f"max_2x2_parity_{idx:02d}"
        prompt = (
            f"Write `{fn}(grid: list[list[int]]) -> int` (`grid` at least 2x2) returning the maximum sum of any `2x2` "
            f"contiguous subgrid whose sum is {par_str}, or `-1` if none exists."
        )
        def ref(grid, _ev=want_even):
            best = None
            for r in range(len(grid) - 1):
                for c in range(len(grid[0]) - 1):
                    s = grid[r][c] + grid[r][c + 1] + grid[r + 1][c] + grid[r + 1][c + 1]
                    if (s % 2 == 0) == _ev: best = s if best is None else max(best, s)
            return -1 if best is None else best
        def mk_g():
            return [[rng.randint(-3, 8) for _ in range(rng.randint(2, 4))] for _ in range(rng.randint(2, 4))]
        def mk_g_rect():
            R, C = rng.randint(2, 4), rng.randint(2, 4)
            return [[rng.randint(-3, 8) for _ in range(C)] for _ in range(R)]
        pub_in = [mk_g_rect(), mk_g_rect()]
        mut_in = [[[x + 2 for x in r] for r in g] for g in pub_in]
        priv_in = [mk_g_rect() for _ in range(6)]
        code = (
            f"def {fn}(grid):\n    best=None\n    for r in range(len(grid)-1):\n        for c in range(len(grid[0])-1):\n"
            f"            s=grid[r][c]+grid[r][c+1]+grid[r+1][c]+grid[r+1][c+1]\n"
            f"            if (s%2==0)=={want_even}: best=s if best is None else max(best,s)\n    return -1 if best is None else best"
        )
        probs.append(make_record(
            f"easy_grid_{idx:02d}", "easy", "subgrid", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    # 13. First-Fit Decreasing bin packing (20)
    for idx in range(20):
        cap = rng.randint(7, 15)
        fn = f"ffd_bins_{idx:02d}"
        prompt = (
            f"Write `{fn}(weights: list[int]) -> int` (non-empty `weights`, each `1 <= w <= {cap}`) simulating "
            f"First-Fit Decreasing bin packing with bin capacity `{cap}`: sort `weights` descending and place each item "
            f"into the FIRST bin with enough remaining capacity (or open a new bin). Return total bins used."
        )
        def ref(weights, _cap=cap):
            bins = []
            for w in sorted(weights, reverse=True):
                for i in range(len(bins)):
                    if bins[i] + w <= _cap: bins[i] += w; break
                else: bins.append(w)
            return len(bins)
        def mk_w(): return [rng.randint(1, cap) for _ in range(rng.randint(4, 10))]
        pub_in = [mk_w(), mk_w()]
        mut_in = [a + [cap] for a in pub_in]
        priv_in = [mk_w() for _ in range(6)]
        code = (
            f"def {fn}(weights):\n    bins=[]\n    for w in sorted(weights,reverse=True):\n"
            f"        for i in range(len(bins)):\n            if bins[i]+w<={cap}: bins[i]+=w; break\n"
            f"        else: bins.append(w)\n    return len(bins)"
        )
        probs.append(make_record(
            f"easy_ffd_{idx:02d}", "easy", "ffd", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv_in],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut_in], code
        ))

    return probs


def build_hard_problems(seed: int = 1337) -> List[Dict]:
    rng = random.Random(seed)
    probs: List[Dict] = []

    # 1. Alternating subsequences with min step difference (20)
    for idx in range(20):
        md = rng.randint(1, 3)
        mod = rng.choice([10**9 + 7, 998244353, 10007])
        fn = f"count_alt_subseq_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (`len(nums) >= 2`) returning the number of subsequences (by index) "
            f"of length `>= 2` that strictly alternate (`a0 < a1 > a2 ...` or `a0 > a1 < a2 ...`) AND satisfy "
            f"`abs(a_{{k+1}} - a_k) >= {md}`, modulo `{mod}`."
        )
        def ref(nums, _md=md, _mod=mod):
            n = len(nums); up, down = [0]*n, [0]*n; ans = 0
            for i in range(n):
                for j in range(i):
                    if nums[i] - nums[j] >= _md: up[i] = (up[i] + 1 + down[j]) % _mod
                    elif nums[j] - nums[i] >= _md: down[i] = (down[i] + 1 + up[j]) % _mod
                ans = (ans + up[i] + down[i]) % _mod
            return ans
        def mk(): return [rng.randint(1, 12) for _ in range(rng.randint(4, 7))]
        pub = [mk(), [3, 3, 3, 3]]
        mut = [[x + 2 for x in pub[0]], [5, 5, 5, 5]]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(nums):\n    n=len(nums); up=[0]*n; down=[0]*n; ans=0\n    for i in range(n):\n        for j in range(i):\n            if nums[i]-nums[j]>={md}: up[i]=(up[i]+1+down[j])%{mod}\n            elif nums[j]-nums[i]>={md}: down[i]=(down[i]+1+up[j])%{mod}\n        ans=(ans+up[i]+down[i])%{mod}\n    return ans"
        probs.append(make_record(
            f"hard_alt_{idx:02d}", "hard", "alt_subseq", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 2. Paint houses: no triple same color & no adjacent forbidden color (20)
    for idx in range(20):
        fc = idx % 3
        fn = f"min_paint_constrained_{idx:02d}"
        prompt = (
            f"Write `{fn}(costs: list[list[int]]) -> int` (`len(costs) >= 2`, each row length 3). Paint `n` houses using colors 0, 1, 2 "
            f"minimizing total cost such that NO THREE consecutive houses share a color AND no two consecutive houses are both color `{fc}`."
        )
        def ref(costs, _fc=fc):
            INF = 10**18; dp = [[INF, INF] for _ in range(3)]
            for c in range(3): dp[c][0] = costs[0][c]
            for row in costs[1:]:
                ndp = [[INF, INF] for _ in range(3)]
                for c in range(3):
                    ndp[c][0] = min(min(dp[p]) for p in range(3) if p != c) + row[c]
                    if c != _fc: ndp[c][1] = dp[c][0] + row[c]
                dp = ndp
            return min(min(x) for x in dp)
        def mk(): return [[rng.randint(1, 10) for _ in range(3)] for _ in range(rng.randint(3, 6))]
        pub = [mk(), mk()]
        mut = [[[x + 1 for x in r] for r in m] for m in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(costs):\n    INF=10**18; dp=[[INF,INF] for _ in range(3)]\n    for c in range(3): dp[c][0]=costs[0][c]\n    for row in costs[1:]:\n        ndp=[[INF,INF] for _ in range(3)]\n        for c in range(3):\n            ndp[c][0]=min(min(dp[p]) for p in range(3) if p!=c)+row[c]\n            if c!={fc}: ndp[c][1]=dp[c][0]+row[c]\n        dp=ndp\n    return min(min(x) for x in dp)"
        probs.append(make_record(
            f"hard_paint_{idx:02d}", "hard", "paint_dp", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 3. Shortest path with cooldown between turbo edges (20)
    for idx in range(20):
        cdk = rng.randint(2, 3)
        fn = f"shortest_turbo_cd_{idx:02d}"
        prompt = (
            f"Write `{fn}(n: int, edges: list[tuple[int, int, int, int]]) -> int` (`n >= 2`). Directed edges are `(u, v, weight, is_turbo)`. "
            f"At least `{cdk}` non-turbo edges (`is_turbo == 0`) must separate any two turbo edges (`is_turbo == 1`). Return min weight `0 -> n-1`, or `-1`."
        )
        def ref(n, edges, _k=cdk):
            adj = [[] for _ in range(n)]
            for u, v, w, t in edges: adj[u].append((v, w, t))
            INF = 10**18; dist = [[INF]*(_k+1) for _ in range(n)]; dist[0][0] = 0; pq = [(0, 0, 0)]
            while pq:
                d, u, cd = heapq.heappop(pq)
                if d != dist[u][cd]: continue
                if u == n - 1: return d
                for v, w, t in adj[u]:
                    if t == 1:
                        if cd > 0: continue
                        ncd = _k
                    else: ncd = max(0, cd - 1)
                    if d + w < dist[v][ncd]: dist[v][ncd] = d + w; heapq.heappush(pq, (d + w, v, ncd))
            return -1
        def mk():
            n = rng.randint(4, 6); edges = []
            for u in range(n - 1):
                edges.append((u, u + 1, rng.randint(1, 3), 1))
                if rng.random() < 0.75: edges.append((u, u + 1, rng.randint(4, 9), 0))
            for _ in range(n):
                u = rng.randint(0, n - 2); v = rng.randint(u + 1, n - 1)
                edges.append((u, v, rng.randint(1, 8), rng.choice([0, 1])))
            return (n, edges)
        pub = [mk(), (3, [(0, 1, 2, 1), (1, 2, 2, 1)])]
        mut = [(pub[0][0], [(u, v, w + 1, t) for u, v, w, t in pub[0][1]]), (3, [(0, 1, 3, 1), (1, 2, 3, 1)])]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(n, edges):\n    import heapq\n    adj=[[] for _ in range(n)]\n    for u,v,w,t in edges: adj[u].append((v,w,t))\n    INF=10**18; dist=[[INF]*{cdk+1} for _ in range(n)]; dist[0][0]=0; pq=[(0,0,0)]\n    while pq:\n        d,u,cd=heapq.heappop(pq)\n        if d!=dist[u][cd]: continue\n        if u==n-1: return d\n        for v,w,t in adj[u]:\n            if t==1:\n                if cd>0: continue\n                ncd={cdk}\n            else: ncd=max(0,cd-1)\n            if d+w<dist[v][ncd]: dist[v][ncd]=d+w; heapq.heappush(pq,(d+w,v,ncd))\n    return -1"
        probs.append(make_record(
            f"hard_turbo_{idx:02d}", "hard", "graph_cd", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 4. Custom-precedence expression evaluator (20)
    for idx in range(20):
        c_div = rng.choice([2, 3]); c_add = rng.randint(0, 3)
        fn = f"eval_custom_expr_{idx:02d}"
        prompt = (
            f"Write `{fn}(expr: str) -> int` evaluating non-negative integers, spaces, `()`, and `+`, `-`, `*`, `#`: "
            f"`+`,`-` have HIGHEST precedence (left-to-right), `*` has MIDDLE precedence (left-to-right), and `#` has LOWEST "
            f"precedence (RIGHT-to-left) with `a # b = abs(a - b) + (a // {c_div}) + {c_add}`."
        )
        def ref(expr, _cd=c_div, _ca=c_add):
            toks, i = [], 0
            while i < len(expr):
                if expr[i].isspace(): i += 1
                elif expr[i] in "+-*#()": toks.append(expr[i]); i += 1
                else:
                    j = i
                    while j < len(expr) and expr[j].isdigit(): j += 1
                    toks.append(int(expr[i:j])); i = j
            pos = [0]
            def ph():
                L = pm()
                if pos[0] < len(toks) and toks[pos[0]] == "#": pos[0] += 1; R = ph(); return abs(L - R) + (L // _cd) + _ca
                return L
            def pm():
                v = pa()
                while pos[0] < len(toks) and toks[pos[0]] == "*": pos[0] += 1; v *= pa()
                return v
            def pa():
                v = pat()
                while pos[0] < len(toks) and toks[pos[0]] in ("+", "-"):
                    op = toks[pos[0]]; pos[0] += 1; r = pat(); v = v + r if op == "+" else v - r
                return v
            def pat():
                if toks[pos[0]] == "(": pos[0] += 1; v = ph(); pos[0] += 1; return v
                v = toks[pos[0]]; pos[0] += 1; return v
            return ph()
        exprs = [
            f"{rng.randint(2,8)} * {rng.randint(2,6)} + {rng.randint(1,5)}",
            f"{rng.randint(8,15)} # {rng.randint(3,7)} # {rng.randint(1,5)}",
            f"{rng.randint(1,6)} + {rng.randint(2,5)} * {rng.randint(2,6)} - {rng.randint(1,3)}",
            f"({rng.randint(2,5)} * {rng.randint(2,5)}) + {rng.randint(1,6)}",
            f"{rng.randint(5,12)} # {rng.randint(1,4)} * {rng.randint(2,4)} + {rng.randint(1,4)}",
            f"({rng.randint(4,9)} # {rng.randint(1,4)}) * {rng.randint(2,4)} - {rng.randint(1,3)}",
            f"{rng.randint(1,5)} # {rng.randint(4,9)} # {rng.randint(1,5)}",
            f"({rng.randint(1,4)} + {rng.randint(2,5)}) # ({rng.randint(2,5)} * {rng.randint(2,4)})",
        ]
        pub, priv = exprs[:2], exprs[2:]
        mut = [f"({pub[0]}) + 1", f"({pub[1]}) + 1"]
        code = f"def {fn}(expr):\n    toks,i=[],0\n    while i<len(expr):\n        if expr[i].isspace(): i+=1\n        elif expr[i] in '+-*#()': toks.append(expr[i]); i+=1\n        else:\n            j=i\n            while j<len(expr) and expr[j].isdigit(): j+=1\n            toks.append(int(expr[i:j])); i=j\n    pos=[0]\n    def ph():\n        L=pm()\n        if pos[0]<len(toks) and toks[pos[0]]=='#': pos[0]+=1; R=ph(); return abs(L-R)+(L//{c_div})+{c_add}\n        return L\n    def pm():\n        v=pa()\n        while pos[0]<len(toks) and toks[pos[0]]=='*': pos[0]+=1; v*=pa()\n        return v\n    def pa():\n        v=pat()\n        while pos[0]<len(toks) and toks[pos[0]] in ('+','-'):\n            op=toks[pos[0]]; pos[0]+=1; r=pat(); v=v+r if op=='+' else v-r\n        return v\n    def pat():\n        if toks[pos[0]]=='(': pos[0]+=1; v=ph(); pos[0]+=1; return v\n        v=toks[pos[0]]; pos[0]+=1; return v\n    return ph()"
        probs.append(make_record(
            f"hard_expr_{idx:02d}", "hard", "expr_parser", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 5. Weighted interval scheduling with gap & max_k cap (20)
    for idx in range(20):
        mk_cap = rng.randint(2, 3)
        fn = f"weighted_iv_gap_k_{idx:02d}"
        prompt = (
            f"Write `{fn}(intervals: list[tuple[int, int, int]], gap: int) -> int` (non-empty `intervals`, `(s, e, v)` with `s < e`). "
            f"Select AT MOST `{mk_cap}` intervals maximizing total value such that any two selected `i, j` (`s_i <= s_j`) satisfy `e_i + gap <= s_j`."
        )
        def ref(ivs, gap, _mk=mk_cap):
            s = sorted(ivs, key=lambda x: x[1]); ends = [x[1] for x in s]; n = len(s)
            dp = [[0]*(_mk+1) for _ in range(n+1)]
            for i, (st, en, val) in enumerate(s, 1):
                p = bisect.bisect_right(ends, st - gap)
                for k in range(1, _mk + 1): dp[i][k] = max(dp[i-1][k], dp[p][k-1] + val)
            return dp[n][_mk]
        def mk():
            ivs = [(a, a + rng.randint(1, 4), rng.randint(2, 15)) for a in [rng.randint(0, 12) for _ in range(rng.randint(4, 7))]]
            return (ivs, rng.randint(1, 3))
        pub = [mk(), mk()]
        mut = [([(s+1, e+1, v+1) for s, e, v in a[0]], a[1]) for a in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(intervals, gap):\n    import bisect\n    s=sorted(intervals,key=lambda x:x[1]); ends=[x[1] for x in s]; n=len(s)\n    dp=[[0]*{mk_cap+1} for _ in range(n+1)]\n    for i,(st,en,val) in enumerate(s,1):\n        p=bisect.bisect_right(ends,st-gap)\n        for k in range(1,{mk_cap+1}): dp[i][k]=max(dp[i-1][k],dp[p][k-1]+val)\n    return dp[n][{mk_cap}]"
        probs.append(make_record(
            f"hard_ivk_{idx:02d}", "hard", "interval_dp", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 6. Restricted Damerau-Levenshtein with custom weights (20)
    for idx in range(20):
        ins, dele, sub, swap = rng.randint(1, 4), rng.randint(2, 5), rng.randint(3, 6), rng.randint(1, 2)
        fn = f"damerau_cost_{idx:02d}"
        prompt = (
            f"Write `{fn}(a: str, b: str) -> int` (non-empty strings `a, b`) returning Restricted Damerau-Levenshtein (OSA) "
            f"distance from `a` to `b` with costs: insert=`{ins}`, delete=`{dele}`, substitute=`{sub}`, adjacent transpose=`{swap}`."
        )
        def ref(a, b, _i=ins, _d=dele, _s=sub, _sw=swap):
            m, n = len(a), len(b); dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1): dp[i][0] = i * _d
            for j in range(1, n+1): dp[0][j] = j * _i
            for i in range(1, m+1):
                for j in range(1, n+1):
                    best = min(dp[i-1][j]+_d, dp[i][j-1]+_i, dp[i-1][j-1]+(0 if a[i-1]==b[j-1] else _s))
                    if i>=2 and j>=2 and a[i-1]==b[j-2] and a[i-2]==b[j-1]: best = min(best, dp[i-2][j-2]+_sw)
                    dp[i][j] = best
            return dp[m][n]
        def mk():
            return ("".join(rng.choice("abcd") for _ in range(rng.randint(3, 6))),
                    "".join(rng.choice("abcd") for _ in range(rng.randint(3, 6))))
        pub = [mk(), mk()]
        mut = [(a + "a", b + "a") for a, b in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(a, b):\n    m,n=len(a),len(b); dp=[[0]*(n+1) for _ in range(m+1)]\n    for i in range(1,m+1): dp[i][0]=i*{dele}\n    for j in range(1,n+1): dp[0][j]=j*{ins}\n    for i in range(1,m+1):\n        for j in range(1,n+1):\n            best=min(dp[i-1][j]+{dele},dp[i][j-1]+{ins},dp[i-1][j-1]+(0 if a[i-1]==b[j-1] else {sub}))\n            if i>=2 and j>=2 and a[i-1]==b[j-2] and a[i-2]==b[j-1]: best=min(best,dp[i-2][j-2]+{swap})\n            dp[i][j]=best\n    return dp[m][n]"
        probs.append(make_record(
            f"hard_osa_{idx:02d}", "hard", "damerau", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 7. Min swaps in k-congruent residue classes (20)
    for idx in range(20):
        k = rng.randint(2, 4)
        fn = f"min_swaps_mod_{idx:02d}"
        prompt = (
            f"Write `{fn}(perm: list[int]) -> int` (permutation of `0..n-1`, `n >= {k+1}`). "
            f"You may swap indices `i, j` ONLY if `i % {k} == j % {k}`. Return minimum swaps to sort ascending, or `-1`."
        )
        def ref(perm, _k=k):
            n, tot = len(perm), 0
            for r in range(min(_k, n)):
                idxs = list(range(r, n, _k)); vals = [perm[i] for i in idxs]
                if sorted(vals) != idxs: return -1
                pos = {v: i for i, v in enumerate(idxs)}; arr = [pos[v] for v in vals]; vis = [False]*len(arr)
                for i in range(len(arr)):
                    if not vis[i]:
                        cyc, cur = 0, i
                        while not vis[cur]: vis[cur] = True; cur = arr[cur]; cyc += 1
                        tot += cyc - 1
            return tot
        def mk(solvable=True):
            n = rng.randint(k + 2, k + 6); p = list(range(n))
            for r in range(k):
                idxs = list(range(r, n, k)); vals = idxs[:]; rng.shuffle(vals)
                for i, v in zip(idxs, vals): p[i] = v
            if not solvable: p[0], p[1] = p[1], p[0]
            return p
        pub = [mk(True), mk(False)]
        mut = [mk(True), mk(False)]
        priv = [mk(i % 3 != 0) for i in range(6)]
        code = f"def {fn}(perm):\n    n,tot=len(perm),0\n    for r in range(min({k},n)):\n        idxs=list(range(r,n,{k})); vals=[perm[i] for i in idxs]\n        if sorted(vals)!=idxs: return -1\n        pos={{v:i for i,v in enumerate(idxs)}}; arr=[pos[v] for v in vals]; vis=[False]*len(arr)\n        for i in range(len(arr)):\n            if not vis[i]:\n                cyc,cur=0,i\n                while not vis[cur]: vis[cur]=True; cur=arr[cur]; cyc+=1\n                tot+=cyc-1\n    return tot"
        probs.append(make_record(
            f"hard_swap_{idx:02d}", "hard", "perm_swaps", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 8. Stone game without repeating opponent's move (20)
    for idx in range(20):
        moves = tuple(sorted(rng.sample([1, 2, 3, 4, 5], 3)))
        fn = f"stone_game_norep_{idx:02d}"
        prompt = (
            f"Write `{fn}(n: int) -> bool` (`n >= 1`). Two players play optimally on `n` stones, removing `m in {list(moves)}` "
            f"(`m <= remaining`), EXCEPT a player cannot pick the exact `m` the opponent just removed on the prior turn. "
            f"No legal move loses. Return True if Player 1 wins."
        )
        def ref(n, _m=moves):
            @lru_cache(maxsize=None)
            def win(rem, forb):
                return any(x <= rem and x != forb and not win(rem - x, x) for x in _m)
            return win(n, -1)
        wins = [x for x in range(1, 25) if ref(x)]
        losses = [x for x in range(1, 25) if not ref(x)] or [1]
        pub = [rng.choice(wins), rng.choice(losses)]
        mut = [rng.choice(wins), rng.choice(losses)]
        priv = [rng.choice(wins if i % 2 == 0 else losses) for i in range(6)]
        code = f"def {fn}(n):\n    from functools import lru_cache\n    @lru_cache(maxsize=None)\n    def win(r,f): return any(x<=r and x!=f and not win(r-x,x) for x in {moves!r})\n    return win(n,-1)"
        probs.append(make_record(
            f"hard_game_{idx:02d}", "hard", "game_dp", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 9. Max bordered subrectangle with minimum side length (20)
    for idx in range(20):
        ms = rng.choice([2, 3])
        fn = f"max_border_rect_{idx:02d}"
        prompt = (
            f"Write `{fn}(grid: list[list[int]]) -> int` (`grid` at least `{ms}x{ms}`) returning the maximum border-only "
            f"perimeter sum of any subrectangle with height `>= {ms}` and width `>= {ms}` (corners once, interior excluded)."
        )
        def ref(grid, _ms=ms):
            R, C = len(grid), len(grid[0]); best = -10**18
            for r1 in range(R):
                for r2 in range(r1 + _ms - 1, R):
                    for c1 in range(C):
                        for c2 in range(c1 + _ms - 1, C):
                            s = sum(grid[r1][c1:c2+1]) + sum(grid[r2][c1:c2+1]) + sum(grid[r][c1] + grid[r][c2] for r in range(r1+1, r2))
                            if s > best: best = s
            return best
        def mk():
            R, C = rng.randint(ms, ms + 2), rng.randint(ms, ms + 2)
            return [[rng.randint(-12, 10) for _ in range(C)] for _ in range(R)]
        pub = [mk(), mk()]
        mut = [[[x + 1 for x in r] for r in g] for g in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(grid):\n    R,C=len(grid),len(grid[0]); best=-10**18\n    for r1 in range(R):\n        for r2 in range(r1+{ms}-1,R):\n            for c1 in range(C):\n                for c2 in range(c1+{ms}-1,C):\n                    s=sum(grid[r1][c1:c2+1])+sum(grid[r2][c1:c2+1])+sum(grid[r][c1]+grid[r][c2] for r in range(r1+1,r2))\n                    if s>best: best=s\n    return best"
        probs.append(make_record(
            f"hard_rect_{idx:02d}", "hard", "border_rect", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 10. Wildcard parentheses with exact max nesting depth (20)
    for idx in range(20):
        md = rng.randint(2, 3)
        fn = f"count_bracket_depth_{idx:02d}"
        prompt = (
            f"Write `{fn}(s: str) -> int` (non-empty `s` of `'('`, `')'`, `'?'`) returning the number of ways to replace "
            f"`'?'` with `'('` or `')'` so `s` is valid parentheses whose maximum nesting depth is EXACTLY `{md}`."
        )
        def ref(s, _md=md):
            dp = {(0, False): 1}
            for ch in s:
                ndp = {}; opts = ("(", ")") if ch == "?" else (ch,)
                for (d, h), c in dp.items():
                    for o in opts:
                        nd = d + (1 if o == "(" else -1)
                        if 0 <= nd <= _md:
                            k = (nd, h or nd == _md); ndp[k] = ndp.get(k, 0) + c
                dp = ndp
            return dp.get((0, True), 0)
        def mk(): return "".join(rng.choices(["?", "(", ")"], weights=[6, 2, 2], k=rng.choice([4, 6, 8])))
        pub = ["?" * (2 * md), mk()]
        mut = ["?" * (2 * md + 2), mk()]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(s):\n    dp={{(0,False):1}}\n    for ch in s:\n        ndp={{}}; opts=('(',')') if ch=='?' else (ch,)\n        for (d,h),c in dp.items():\n            for o in opts:\n                nd=d+(1 if o=='(' else -1)\n                if 0<=nd<={md}: k=(nd,h or nd=={md}); ndp[k]=ndp.get(k,0)+c\n        dp=ndp\n    return dp.get((0,True),0)"
        probs.append(make_record(
            f"hard_brk_{idx:02d}", "hard", "bracket_dp", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 11. Min partitions into distinct-element subarrays with XOR <= limit (20)
    for idx in range(20):
        lim = rng.randint(3, 7)
        fn = f"min_xor_parts_{idx:02d}"
        prompt = (
            f"Write `{fn}(nums: list[int]) -> int` (non-empty `nums`) partitioning `nums` into minimum contiguous non-empty "
            f"subarrays where EVERY subarray has bitwise XOR `<= {lim}` AND no duplicate elements (`-1` if impossible)."
        )
        def ref(nums, _lim=lim):
            n, INF = len(nums), 10**9; dp = [INF]*(n+1); dp[0] = 0
            for i in range(n):
                if dp[i] == INF: continue
                seen, xr = set(), 0
                for j in range(i, n):
                    if nums[j] in seen: break
                    seen.add(nums[j]); xr ^= nums[j]
                    if xr <= _lim and dp[i]+1 < dp[j+1]: dp[j+1] = dp[i]+1
            return -1 if dp[n] == INF else dp[n]
        def mk(): return [rng.randint(1, lim + 3) for _ in range(rng.randint(4, 8))]
        pub = [mk(), [lim + 1, lim + 1]]
        mut = [pub[0][::-1], [lim + 2, lim + 2]]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(nums):\n    n,INF=len(nums),10**9; dp=[INF]*(n+1); dp[0]=0\n    for i in range(n):\n        if dp[i]==INF: continue\n        seen,xr=set(),0\n        for j in range(i,n):\n            if nums[j] in seen: break\n            seen.add(nums[j]); xr^=nums[j]\n            if xr<={lim} and dp[i]+1<dp[j+1]: dp[j+1]=dp[i]+1\n    return -1 if dp[n]==INF else dp[n]"
        probs.append(make_record(
            f"hard_xor_{idx:02d}", "hard", "xor_part", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 12. Knight shortest path with 1-use portal network (20)
    for idx in range(20):
        bn = rng.randint(5, 7)
        fn = f"knight_portal_{idx:02d}"
        prompt = (
            f"Write `{fn}(start: tuple[int, int], target: tuple[int, int], portals: list[tuple[int, int]]) -> int` "
            f"on a `{bn}x{bn}` board (`0 <= r, c < {bn}`). Knight moves cost 1; standing on any portal lets you teleport to "
            f"any other portal for cost 1 (at most ONCE total). Return min moves or `-1`."
        )
        def ref(st, tg, portals, _n=bn):
            if st == tg: return 0
            pset = set(portals); mv = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
            q = deque([(st[0], st[1], 0, 0)]); vis = {(st[0], st[1], 0)}; pexp = False
            while q:
                r, c, u, d = q.popleft()
                for dr, dc in mv:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<_n and 0<=nc<_n and (nr,nc,u) not in vis:
                        if (nr, nc) == tg: return d + 1
                        vis.add((nr,nc,u)); q.append((nr,nc,u,d+1))
                if u == 0 and (r, c) in pset and not pexp:
                    pexp = True
                    for pr, pc in pset:
                        if (pr, pc) != (r, c) and (pr, pc, 1) not in vis:
                            if (pr, pc) == tg: return d + 1
                            vis.add((pr, pc, 1)); q.append((pr, pc, 1, d+1))
            return -1
        def mk():
            st = (rng.randint(0, bn-1), rng.randint(0, bn-1))
            tg = (rng.randint(0, bn-1), rng.randint(0, bn-1))
            ps = [(rng.randint(0, bn-1), rng.randint(0, bn-1)) for _ in range(rng.randint(2, 3))]
            return (st, tg, ps)
        pub = [((0, 0), (bn-1, bn-1), [(1, 2), (bn-2, bn-3)]), mk()]
        mut = [((0, bn-1), (bn-1, 0), [(1, bn-3), (bn-2, 2)]), mk()]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(start, target, portals):\n    from collections import deque\n    if start==target: return 0\n    pset=set(portals); mv=[(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]\n    q=deque([(start[0],start[1],0,0)]); vis={{(start[0],start[1],0)}}; pexp=False\n    while q:\n        r,c,u,d=q.popleft()\n        for dr,dc in mv:\n            nr,nc=r+dr,c+dc\n            if 0<=nr<{bn} and 0<=nc<{bn} and (nr,nc,u) not in vis:\n                if (nr,nc)==target: return d+1\n                vis.add((nr,nc,u)); q.append((nr,nc,u,d+1))\n        if u==0 and (r,c) in pset and not pexp:\n            pexp=True\n            for pr,pc in pset:\n                if (pr,pc)!=(r,c) and (pr,pc,1) not in vis:\n                    if (pr,pc)==target: return d+1\n                    vis.add((pr,pc,1)); q.append((pr,pc,1,d+1))\n    return -1"
        probs.append(make_record(
            f"hard_knt_{idx:02d}", "hard", "knight_bfs", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 13. Stock profit with fee, 1-day cooldown, and at most k transactions (20)
    for idx in range(20):
        fee = rng.randint(1, 3); k = rng.randint(2, 3)
        fn = f"stock_cool_fee_{idx:02d}"
        prompt = (
            f"Write `{fn}(prices: list[int]) -> int` (`len(prices) >= 2`). Complete at most `{k}` buy-sell transactions "
            f"(max 1 share held), where each sell incurs fee `{fee}` AND a 1-day cooldown before the next buy. Return max net profit."
        )
        def ref(prices, _fee=fee, _k=k):
            NEG = -10**18; h, f, c = [NEG]*(_k+1), [NEG]*(_k+1), [NEG]*(_k+1); f[0] = 0
            for p in prices:
                nh = list(h); nf = [max(f[t], c[t]) for t in range(_k+1)]; nc = [NEG]*(_k+1)
                for t in range(_k+1):
                    if f[t] != NEG: nh[t] = max(nh[t], f[t] - p)
                    if t < _k and h[t] != NEG: nc[t+1] = max(nc[t+1], h[t] + p - _fee)
                h, f, c = nh, nf, nc
            return max(max(f), max(c))
        def mk(): return [rng.randint(1, 15) for _ in range(rng.randint(5, 9))]
        pub = [mk(), mk()]
        mut = [[x + 2 for x in a] for a in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(prices):\n    NEG=-10**18; h=[NEG]*{k+1}; f=[NEG]*{k+1}; c=[NEG]*{k+1}; f[0]=0\n    for p in prices:\n        nh=list(h); nf=[max(f[t],c[t]) for t in range({k+1})]; nc=[NEG]*{k+1}\n        for t in range({k+1}):\n            if f[t]!=NEG: nh[t]=max(nh[t],f[t]-p)\n            if t<{k} and h[t]!=NEG: nc[t+1]=max(nc[t+1],h[t]+p-{fee})\n        h,f,c=nh,nf,nc\n    return max(max(f),max(c))"
        probs.append(make_record(
            f"hard_stk_{idx:02d}", "hard", "stock_dp", fn, prompt,
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in pub],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in priv],
            [f"assert {fn}({a!r}) == {ref(a)!r}" for a in mut], code
        ))

    # 14. LCS avoiding forbidden contiguous substring (20)
    for idx in range(20):
        forb = rng.choice(["aba", "bab", "aab", "bba", "abc", "cab"])
        fn = f"lcs_avoid_{idx:02d}"
        prompt = (
            f"Write `{fn}(a: str, b: str) -> int` (non-empty strings over `'a'..'c'`) returning the length of the longest "
            f"common subsequence `z` of `a` and `b` NOT containing `{forb!r}` as a contiguous substring."
        )
        def ref(a, b, _fb=forb):
            m, n, L = len(a), len(b), len(_fb); pi = [0]*L
            for i in range(1, L):
                j = pi[i-1]
                while j > 0 and _fb[i] != _fb[j]: j = pi[j-1]
                if _fb[i] == _fb[j]: j += 1
                pi[i] = j
            tr = {}
            for st in range(L):
                for ch in "abc":
                    j = st
                    while j > 0 and _fb[j] != ch: j = pi[j-1]
                    if _fb[j] == ch: j += 1
                    tr[(st, ch)] = j
            NEG = -10**9; dp = [[[NEG]*L for _ in range(n+1)] for _ in range(m+1)]; dp[0][0][0] = 0
            for i in range(m+1):
                for j in range(n+1):
                    for st in range(L):
                        v = dp[i][j][st]
                        if v == NEG: continue
                        if i < m and v > dp[i+1][j][st]: dp[i+1][j][st] = v
                        if j < n and v > dp[i][j+1][st]: dp[i][j+1][st] = v
                        if i < m and j < n and a[i] == b[j]:
                            ns = tr[(st, a[i])]
                            if ns < L and v+1 > dp[i+1][j+1][ns]: dp[i+1][j+1][ns] = v+1
            return max(dp[m][n])
        def mk():
            return ("".join(rng.choice("abc") for _ in range(rng.randint(5, 8))),
                    "".join(rng.choice("abc") for _ in range(rng.randint(5, 8))))
        pub = [(forb + forb[:2], forb + forb[:2]), mk()]
        mut = [(a + "c", b + "c") for a, b in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(a, b):\n    fb={forb!r}; m,n,L=len(a),len(b),len(fb); pi=[0]*L\n    for i in range(1,L):\n        j=pi[i-1]\n        while j>0 and fb[i]!=fb[j]: j=pi[j-1]\n        if fb[i]==fb[j]: j+=1\n        pi[i]=j\n    tr={{}}\n    for st in range(L):\n        for ch in set(a)|set(b):\n            j=st\n            while j>0 and fb[j]!=ch: j=pi[j-1]\n            if fb[j]==ch: j+=1\n            tr[(st,ch)]=j\n    NEG=-10**9; dp=[[[NEG]*L for _ in range(n+1)] for _ in range(m+1)]; dp[0][0][0]=0\n    for i in range(m+1):\n        for j in range(n+1):\n            for st in range(L):\n                v=dp[i][j][st]\n                if v==NEG: continue\n                if i<m and v>dp[i+1][j][st]: dp[i+1][j][st]=v\n                if j<n and v>dp[i][j+1][st]: dp[i][j+1][st]=v\n                if i<m and j<n and a[i]==b[j]:\n                    ns=tr[(st,a[i])]\n                    if ns<L and v+1>dp[i+1][j+1][ns]: dp[i+1][j+1][ns]=v+1\n    return max(dp[m][n])"
        probs.append(make_record(
            f"hard_lcs_{idx:02d}", "hard", "lcs_kmp", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 15. Constrained topological orderings (bitmask DP) (20)
    for idx in range(20):
        fn = f"count_topo_forb_{idx:02d}"
        prompt = (
            f"Write `{fn}(n: int, edges: list[tuple[int, int]], forbidden_adj: list[tuple[int, int]]) -> int` (`n >= 3`). "
            f"Return the number of topological orderings of `0..n-1` respecting prerequisites `edges` where no `(x, y)` in `forbidden_adj` appear consecutively."
        )
        def ref(n, edges, forb):
            pr = [0]*n
            for u, v in edges: pr[v] |= (1 << u)
            fb = set(forb); dp = [[0]*n for _ in range(1 << n)]
            for u in range(n):
                if pr[u] == 0: dp[1 << u][u] = 1
            for mask in range(1, 1 << n):
                for last in range(n):
                    c = dp[mask][last]
                    if not c: continue
                    for nxt in range(n):
                        if not (mask & (1 << nxt)) and (mask & pr[nxt]) == pr[nxt] and (last, nxt) not in fb:
                            dp[mask | (1 << nxt)][nxt] += c
            return sum(dp[(1 << n) - 1])
        def mk():
            n = rng.randint(4, 6)
            edges = [(u, v) for u in range(n) for v in range(u+1, n) if rng.random() < 0.35]
            forb = list({(x, y) for x, y in ((rng.randint(0, n-1), rng.randint(0, n-1)) for _ in range(rng.randint(1, 3))) if x != y})
            return (n, edges, forb)
        pub = [mk(), mk()]
        mut = [(a[0], a[1], a[2][::-1]) for a in pub]
        priv = [mk() for _ in range(6)]
        code = f"def {fn}(n, edges, forbidden_adj):\n    pr=[0]*n\n    for u,v in edges: pr[v]|=(1<<u)\n    fb=set(forbidden_adj); dp=[[0]*n for _ in range(1<<n)]\n    for u in range(n):\n        if pr[u]==0: dp[1<<u][u]=1\n    for mask in range(1,1<<n):\n        for last in range(n):\n            c=dp[mask][last]\n            if not c: continue\n            for nxt in range(n):\n                if not(mask&(1<<nxt)) and (mask&pr[nxt])==pr[nxt] and (last,nxt) not in fb: dp[mask|(1<<nxt)][nxt]+=c\n    return sum(dp[(1<<n)-1])"
        probs.append(make_record(
            f"hard_topo_{idx:02d}", "hard", "topo_dp", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    # 16. Tree two-color independent set with index-direction asymmetry (20)
    for idx in range(20):
        fd = (idx % 2 == 1)
        rule = "u_red > v_blue" if fd else "u_red < v_blue"
        fn = f"tree_two_color_{idx:02d}"
        prompt = (
            f"Write `{fn}(n: int, edges: list[tuple[int, int]], w1: list[int], w2: list[int]) -> int` (`n >= 2`, undirected tree `0..n-1`). "
            f"Color each node uncolored (0), Red (`w1[u]`), or Blue (`w2[u]`) so adjacent nodes never share a color AND Red adjacent to Blue requires `{rule}`. Return max value."
        )
        def ref(n, edges, w1, w2, _fd=fd):
            adj = [[] for _ in range(n)]
            for u, v in edges: adj[u].append(v); adj[v].append(u)
            par, order = [-1]*n, [0]
            for u in order:
                for v in adj[u]:
                    if v != par[u]: par[v] = u; order.append(v)
            dp = [[0, w1[u], w2[u]] for u in range(n)]
            def ok(u, cu, v, cv):
                if cu == 0 or cv == 0: return True
                if cu == cv: return False
                if cu == 1 and cv == 2: return (u > v) if _fd else (u < v)
                return (v > u) if _fd else (v < u)
            for u in reversed(order):
                for v in adj[u]:
                    if par[v] == u:
                        for cu in range(3): dp[u][cu] += max(dp[v][cv] for cv in range(3) if ok(u, cu, v, cv))
            return max(dp[0])
        def mk():
            n = rng.randint(4, 7)
            edges = [(rng.randint(0, i-1), i) for i in range(1, n)]
            return (n, edges, [rng.randint(1, 12) for _ in range(n)], [rng.randint(1, 12) for _ in range(n)])
        pub = [mk(), mk()]
        mut = [(a[0], a[1], [x+1 for x in a[2]], [x+1 for x in a[3]]) for a in pub]
        priv = [mk() for _ in range(6)]
        cmp_expr = "(u>v) if cu==1 else (v>u)" if fd else "(u<v) if cu==1 else (v<u)"
        code = f"def {fn}(n, edges, w1, w2):\n    adj=[[] for _ in range(n)]\n    for u,v in edges: adj[u].append(v); adj[v].append(u)\n    par,order=[-1]*n,[0]\n    for u in order:\n        for v in adj[u]:\n            if v!=par[u]: par[v]=u; order.append(v)\n    dp=[[0,w1[u],w2[u]] for u in range(n)]\n    def ok(u,cu,v,cv):\n        if cu==0 or cv==0: return True\n        if cu==cv: return False\n        return {cmp_expr}\n    for u in reversed(order):\n        for v in adj[u]:\n            if par[v]==u:\n                for cu in range(3): dp[u][cu]+=max(dp[v][cv] for cv in range(3) if ok(u,cu,v,cv))\n    return max(dp[0])"
        probs.append(make_record(
            f"hard_tree_{idx:02d}", "hard", "tree_dp", fn, prompt,
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}, {a[3]!r}) == {ref(*a)!r}" for a in pub],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}, {a[3]!r}) == {ref(*a)!r}" for a in priv],
            [f"assert {fn}({a[0]!r}, {a[1]!r}, {a[2]!r}, {a[3]!r}) == {ref(*a)!r}" for a in mut], code
        ))

    return probs


def stratified_split(probs: List[Dict], n_eval: int = 40, seed: int = 42):
    by_fam = defaultdict(list)
    for p in probs:
        by_fam[p["family"]].append(p)
    rng = random.Random(seed)
    train, eval_set = [], []
    fams = sorted(by_fam.keys())
    per_fam = n_eval // len(fams)
    rem = n_eval % len(fams)
    for i, fam in enumerate(fams):
        items = by_fam[fam][:]
        rng.shuffle(items)
        take = per_fam + (1 if i < rem else 0)
        eval_set.extend(items[:take])
        train.extend(items[take:])
    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    out_dir = scratch_dir / "data" / "rl"

    easy = build_easy_problems(seed=42)
    hard = build_hard_problems(seed=1337)
    easy_tr, easy_ev = stratified_split(easy, n_eval=40, seed=42)
    hard_tr, hard_ev = stratified_split(hard, n_eval=40, seed=1337)

    write_jsonl(out_dir / "train_easy.jsonl", easy_tr)
    write_jsonl(out_dir / "eval_easy.jsonl", easy_ev)
    write_jsonl(out_dir / "train_hard.jsonl", hard_tr)
    write_jsonl(out_dir / "eval_hard.jsonl", hard_ev)

    summary = {
        "easy_train": len(easy_tr), "easy_eval": len(easy_ev),
        "hard_train": len(hard_tr), "hard_eval": len(hard_ev),
        "easy_families": len({p["family"] for p in easy}),
        "hard_families": len({p["family"] for p in hard}),
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
