from functions import get_primitive_int, generate_gf_tables, get_anf, get_canonical_form, get_cubic_indices
from mults import calculate_mult
from numba import njit # just-in-time compiler
import numpy as np
from numba_progress import ProgressBar
import math
import csv
import os # to check if csv files already exist

@njit(nogil=True)
def get_uniformity_part(N, sbox, pbar): # calculates if f(x) = xTr(ax^[2^j + 2^(2j)]) has optimal delta^2
    dddt = np.zeros(N, dtype=np.int32)
    for a in range(1, N):
        for b in range(a+1, N):
            ab = a ^ b
            if b < ab:
                for k in range(N):
                    dddt[k] = 0
                for x in range(N):
                    c = sbox[x] ^ sbox[x^a] ^ sbox[x^b] ^ sbox[x^ab]
                    dddt[c] += 1
                    if dddt[c] > 4:
                        return 0
                pbar.update()

    return 4

@njit
def get_tt(N, M, MOD, j, exp_table, log_table): # computes the truth table for the function f(x) = x^d1 + a^i x^d2
    tt = np.zeros(N, dtype=np.uint64)

    for x in range(1, N): # calculates sbox
        idx = log_table[x] # x exponent
        idx2 = (idx*(2**j + 2**(2*j)) + 1) % MOD # calculates t = ax^[2^j + 2^(2j)] exponent
        trace = exp_table[idx2] ^ exp_table[(idx2*(2**M)) % MOD] # calculates Tr(t) = t + t^(2^m)
        tt[x] = exp_table[(idx + log_table[trace]) % MOD] # calculates xTr(t)
        
    return tt

def calculate_hist(m, tt, cubic_indices):
    anf = get_anf(n, tt) # gets the ANF of the truth table

    # gets cubic coefficients of the ANF
    cubic_coeffs = np.zeros(m, dtype=np.int64)
    for i in range(m):
        cubic_coeffs[i] = anf[cubic_indices[i]]
        
    gPrime = get_canonical_form(n, m, cubic_coeffs) # converts cubic coeffs into canonical form
    
    if gPrime[0] > 0: # if the canonical form exists
        zero_count, hits = calculate_mult(n, gPrime) # calculates the multiplicity spectrum
        unique_hits, counts_hits = np.unique(hits, return_counts=True)

        # formats, prints and saves output in textfile
        out_parts = []
        if zero_count > 0:
            out_parts.append(f"0:{zero_count}")
        for u, c in zip(unique_hits, counts_hits):
            out_parts.append(f"{u}:{c}")

        return '|'.join(out_parts)

def main(n):
    N = 2**n
    M = int(n/2)
    MOD = N-1
    m = int((n*(n-1)*(n-2))/6)
    count = int((N-1)*(N-2)/6) # number of (a, b)-pairs to be checked

    if not os.path.exists('optimal_infinites.csv'): # writes headers if file does not exist
        with open('optimal_infinites.csv', 'a', newline='') as f:
            csv.writer(f).writerow(['n', 'j', 'delta2', 'mult'])

    poly = get_primitive_int(n)
    exp_table, log_table = generate_gf_tables(N, MOD, poly)
    cubic_indices = get_cubic_indices(n, m)

    for j in range(1, n):
        if math.gcd(j, n) == 1:
            tt = get_tt(N, M, MOD, j, exp_table, log_table) # gets truth table
            with ProgressBar(total=count, desc='j = ' + str(j)) as pbar:
                delta2 = get_uniformity_part(N, tt, pbar)
            if delta2 == 4:
                mult = calculate_hist(m, tt, cubic_indices)
                with open('optimal_infinites.csv', 'a', newline='') as f:
                    csv.writer(f).writerow([n, j, delta2, mult])

if __name__ == "__main__":
    ns = [6, 10, 14, 22]
    for n in ns:
        print('n =', n)
        main(n)
        print()