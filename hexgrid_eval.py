#!/usr/bin/env python3
"""
Reference implementation and experimental evaluation for the
hexagonal-grid graph-labelling encryption scheme (Z_6, triple labels).

Reproduces the numbers reported in the "Experimental Evaluation" section:
  E1  correctness / round-trip            (unique recovery, G4)
  E2  runtime scaling vs n                 (O(n^2) claim, Sec. complexity)
  E3  key avalanche                        (affine vs PRF mask, Sec. avalanche)
  E4  known-plaintext attack               (affine breakable, PRF resists, Sec. kpa)
  E5  ciphertext uniformity (chi-square)   (indistinguishability, Prop. indist)
  E6  key-space growth table               (Thm. bound)

All randomness is seeded for reproducibility.
"""

import hashlib, time, random
import numpy as np
from scipy import stats
from sympy import nextprime

SEED = 20260611
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Masking functions
# --------------------------------------------------------------------------
def F_affine(k, r, pos, M):
    return (k * pos + r) % M

def F_prf(k, r, pos, M):
    h = hashlib.sha256(f"{k}|{r}|{pos}".encode()).digest()
    return int.from_bytes(h, "big") % M

# --------------------------------------------------------------------------
# Decomposition: Num = sum_{i=0}^5 Num_i + 3, Num_i = i (mod 6), Num_i >= 6
# (clean six-residue case, Num = 0 (mod 6); matches Example 1)
# --------------------------------------------------------------------------
def decompose(Num, rng):
    assert Num % 6 == 0 and Num >= 48
    base = [6, 7, 8, 9, 10, 11]              # residues 0..5
    extra_units = (Num - 3 - sum(base)) // 6  # number of "+6" units to spread
    comps = base[:]
    # multinomial split of extra_units across the 6 components (no per-unit loop)
    cuts = sorted(rng.randrange(extra_units + 1) for _ in range(5))
    bounds = [0] + cuts + [extra_units]
    for i in range(6):
        comps[i] += 6 * (bounds[i + 1] - bounds[i])
    assert sum(comps) + 3 == Num
    for i in range(6):
        assert comps[i] % 6 == i and comps[i] >= 6
    return comps  # Num_0..Num_5

# --------------------------------------------------------------------------
# Core encode/decode of the C_6 (the secret-carrying cycle)
# --------------------------------------------------------------------------
def encrypt_cycle(Num, k, r, M, Fmask, rng):
    comps = decompose(Num, rng)
    masked, residues = [], []
    for i in range(6):
        b = comps[i] // 6
        bstar = (b + Fmask(k, r, i + 1, M)) % M
        masked.append(bstar)
        residues.append(comps[i] % 6)       # = i, carried by head-vertex tag l_j
    return masked, residues

def decrypt_cycle(masked, residues, k, r, M, Fmask):
    Num = 3
    for i in range(6):
        b = (masked[i] - Fmask(k, r, i + 1, M)) % M
        Num += 6 * b + residues[i]
    return Num

# --------------------------------------------------------------------------
# E1: correctness / round-trip over random instances
# --------------------------------------------------------------------------
def exp_correctness(trials=20_000):
    rng = random.Random(SEED)
    ok_aff = ok_prf = 0
    for _ in range(trials):
        Num = rng.randrange(48, 10_000_000) // 6 * 6
        k = rng.randrange(100, 10_000)
        r = rng.randrange(0, 10_000)
        M = nextprime(Num // 6)
        for Fm, counter in ((F_affine, "aff"), (F_prf, "prf")):
            m, res = encrypt_cycle(Num, k, r, M, Fm, rng)
            dec = decrypt_cycle(m, res, k, r, M, Fm)
            if dec == Num:
                if counter == "aff": ok_aff += 1
                else: ok_prf += 1
    return ok_aff, ok_prf, trials

# --------------------------------------------------------------------------
# E2: runtime scaling vs n  (full-grid label assignment, O(1) per element)
# --------------------------------------------------------------------------
def label_full_grid(n, k, r, M):
    """Assign vertex tag (l+m)%6 to 2n^2-2 vertices and edge tag
    (k+b*+c)%6 to 3n^2-2n-2 edges; O(1) work each -> O(n^2) total."""
    V = 2 * n * n - 2
    E = 3 * n * n - 2 * n - 2
    s = 0
    for m in range(1, V + 1):
        l = (m - 1) % 6
        s ^= (l + m) % 6                     # vertex tag x
    for e in range(E):
        bstar = (e * 1315423911) % M         # cheap deterministic stand-in
        c = (k - (e % 7) - (e % 5))
        s ^= (k + bstar + c) % 6             # edge tag y
    return s

def exp_runtime(ns=(6, 12, 18, 24, 30, 36, 48, 60), reps=5):
    k, r, M = 4321, 7, nextprime(10_000)
    rows = []
    for n in ns:
        best = min(_time(lambda: label_full_grid(n, k, r, M)) for _ in range(reps))
        rows.append((n, 2 * n * n - 2, 3 * n * n - 2 * n - 2, best))
    return rows

def _time(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0

# --------------------------------------------------------------------------
# E3: key avalanche  (k -> k+1)
# --------------------------------------------------------------------------
def exp_avalanche(trials=20_000):
    rng = random.Random(SEED + 1)
    aff_comp_changed = []      # affine: how many of 6 masked components change
    prf_bits = []              # PRF: fraction of output bits flipped
    for _ in range(trials):
        Num = rng.randrange(48, 5_000_000) // 6 * 6
        k = rng.randrange(100, 10_000); r = rng.randrange(0, 10_000)
        M = nextprime(Num // 6)
        # affine: component-level change count
        m0, _ = encrypt_cycle(Num, k,     r, M, F_affine, random.Random(1))
        m1, _ = encrypt_cycle(Num, k + 1, r, M, F_affine, random.Random(1))
        aff_comp_changed.append(sum(a != b for a, b in zip(m0, m1)))
        # PRF: bit-level avalanche of the 6-component masked vector
        p0, _ = encrypt_cycle(Num, k,     r, M, F_prf, random.Random(1))
        p1, _ = encrypt_cycle(Num, k + 1, r, M, F_prf, random.Random(1))
        width = max(M.bit_length(), 1)
        b0 = "".join(format(x, f"0{width}b") for x in p0)
        b1 = "".join(format(x, f"0{width}b") for x in p1)
        flips = sum(c0 != c1 for c0, c1 in zip(b0, b1))
        prf_bits.append(flips / len(b0))
    return (np.mean(aff_comp_changed), np.min(aff_comp_changed),
            np.mean(prf_bits), np.std(prf_bits))

# --------------------------------------------------------------------------
# E4: known-plaintext attack on affine mask; PRF resistance
# --------------------------------------------------------------------------
def exp_kpa(trials=10_000):
    rng = random.Random(SEED + 2)
    aff_recovered = prf_recovered = 0
    for _ in range(trials):
        M = nextprime(rng.randrange(2000, 50_000))
        k = rng.randrange(1, M); r = rng.randrange(0, M)
        p1, p2 = 1, 2
        b1, b2 = rng.randrange(0, M), rng.randrange(0, M)
        # ---- affine: two known plaintexts -> closed-form key recovery
        bs1 = (b1 + F_affine(k, r, p1, M)) % M
        bs2 = (b2 + F_affine(k, r, p2, M)) % M
        lhs = ((bs1 - b1) - (bs2 - b2)) % M
        inv = pow((p1 - p2) % M, -1, M)
        k_rec = (lhs * inv) % M
        if k_rec == k % M:
            aff_recovered += 1
        # ---- PRF: same linear formula must NOT recover the key
        bs1 = (b1 + F_prf(k, r, p1, M)) % M
        bs2 = (b2 + F_prf(k, r, p2, M)) % M
        lhs = ((bs1 - b1) - (bs2 - b2)) % M
        k_rec = (lhs * inv) % M
        if k_rec == k % M:
            prf_recovered += 1
    return aff_recovered, prf_recovered, trials

# --------------------------------------------------------------------------
# E5: ciphertext uniformity under PRF mask (chi-square goodness of fit)
# --------------------------------------------------------------------------
def exp_uniformity(samples=120_000, bins=64):
    rng = random.Random(SEED + 3)
    M = nextprime(2_000_003)
    k = rng.randrange(1, M)
    vals = []
    for _ in range(samples):
        r = rng.randrange(0, M)
        pos = rng.randrange(1, 7)
        b = rng.randrange(0, M // 6)
        vals.append(((b + F_prf(k, r, pos, M)) % M))
    counts, _ = np.histogram(vals, bins=bins, range=(0, M))
    expected = np.full(bins, samples / bins)
    chi2, p = stats.chisquare(counts, expected)
    return chi2, p, bins, samples

# --------------------------------------------------------------------------
# E6: key-space growth
# --------------------------------------------------------------------------
def exp_keyspace(ns=(6, 12, 18, 24, 30, 48)):
    rows = []
    for n in ns:
        K = 7 * n * n - 2 * n - 10
        cycles = (n - 1) ** 2
        rows.append((n, K, cycles, K * cycles))
    return rows

# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("E1  Correctness / round-trip")
    a, p, t = exp_correctness()
    print(f"  affine: {a}/{t} recovered ; PRF: {p}/{t} recovered")

    print("=" * 60)
    print("E2  Runtime scaling vs n")
    rt = exp_runtime()
    for n, V, E, sec in rt:
        print(f"  n={n:3d}  V={V:6d}  E={E:6d}  time={sec*1e3:8.3f} ms")
    # log-log slope
    ns = np.array([row[0] for row in rt], float)
    ts = np.array([row[3] for row in rt], float)
    slope = np.polyfit(np.log(ns), np.log(ts), 1)[0]
    print(f"  empirical log-log slope (expect ~2.0): {slope:.3f}")

    print("=" * 60)
    print("E3  Key avalanche")
    am, amin, pm, ps = exp_avalanche()
    print(f"  affine: mean components changed = {am:.4f}/6 (min {amin})")
    print(f"  PRF   : mean output-bit flip = {pm*100:.2f}%  (sd {ps*100:.2f}%)")

    print("=" * 60)
    print("E4  Known-plaintext attack")
    ar, pr, t = exp_kpa()
    print(f"  affine key recovered: {ar}/{t} ({100*ar/t:.1f}%)")
    print(f"  PRF    key recovered: {pr}/{t} ({100*pr/t:.4f}%)")

    print("=" * 60)
    print("E5  Ciphertext uniformity (PRF mask, chi-square)")
    chi2, pval, bins, ns_ = exp_uniformity()
    print(f"  chi2={chi2:.2f}  dof={bins-1}  p-value={pval:.4f}  (n={ns_})")

    print("=" * 60)
    print("E6  Key-space growth")
    for n, K, cyc, tot in exp_keyspace():
        print(f"  n={n:3d}  |K|={K:8d}  cycles={cyc:6d}  product={tot:.3e}")
