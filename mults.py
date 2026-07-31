from functions import *
import numpy as np
from numba import njit, types
from numba.typed import Dict
from configparser import ConfigParser # to read config file
import time
import os
import csv

@njit
def calculate_mult(n, coeffs): # calculates multiplicities by mapping 2D subspaces to their kernels and counting the collisions
    kernel_counts = Dict.empty(key_type=types.uint64, value_type=types.int64)
    count = 0

    # iterates through every possible 2D subspace defined by basis vectors (a, b)
    for a in range(1, 1 << n):
        for ab in range(a + 1, 1 << n):
            b = a ^ ab
            if ab < b: # ensures we only process each unique {a, b, a^b} set exactly once
                count += 1
                M = np.zeros(n, dtype=np.int64)
                c = 0
                
                # builds the matrix M for the 2D subspace
                for i in range(n):
                    ai = (a >> i) & 1
                    bi = (b >> i) & 1
                    for j in range(i + 1, n):
                        aj = (a >> j) & 1
                        bj = (b >> j) & 1
                        for k in range(j + 1, n):
                            ak = (a >> k) & 1
                            bk = (b >> k) & 1

                            # calculates the derivative terms
                            term_a = (aj & bk) ^ (ak & bj)
                            term_b = (ai & bk) ^ (ak & bi)
                            term_c = (ai & bj) ^ (aj & bi)

                            # if a term is 1, XOR the corresponding cubic coefficient into M
                            if term_a:
                                M[i] ^= coeffs[c]
                            if term_b:
                                M[j] ^= coeffs[c]
                            if term_c:
                                M[k] ^= coeffs[c]
                            c += 1

                # finds the kernel and counts the no. of collisions
                K = get_kernel_basis(n, M)
                if K[1] != 0: # if the kernel has dimension >2
                    key = (np.uint64(K[1]) << 32) | np.uint64(K[2]) # packs the two smallest kenerel basis vectors into a single integer
                    kernel_counts[key] = kernel_counts.get(key, np.int64(0)) + np.int64(1) # increments the multiplicity count for this specific kernel image

    zero_count = count - len(kernel_counts) # any subspace that didn't map to a valid 2D kernel is counted as a '0' mapping

    # extracts just the collision counts (multiplicities) into an array
    hits = np.zeros(len(kernel_counts), dtype=np.int64)
    idx = 0
    for val in kernel_counts.values():
        hits[idx] = val
        idx += 1
        
    return zero_count, hits

def main(n):
    start = time.time()
    print(f'\n--- Starting Analysis for n = {n} ---')

    N = 2**n
    MOD = 2**n - 1
    m = int((n*(n-1)*(n-2))/6)
    poly = get_primitive_int(n)
    exp_t, log_t = generate_gf_tables(N, MOD, poly)
    
    cubic_indices = [] # pre-computes these
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cubic_indices.append((1<<i) | (1<<j) | (1<<k))
    cubic_indices = np.array(cubic_indices, dtype=np.int64)

    exps = get_exponents(n, MOD)
    
    for d in exps:
        tt = get_tt(N, d[0], d[1], d[2], exp_t, log_t, MOD) # gets truth table
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
                
            hist = '|'.join(out_parts)
            print(f"Exponents {d.tolist()}  =>  {hist}")

            with open('mults.csv', 'a+', newline='') as f:
                csv.writer(f).writerow([n, d[0], d[1], d[2], hist])
            
    end = time.time()
    print('Time taken:', round(end-start, 2))

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('mults', 'min')
    user_max = config.get('mults', 'max')

    try:
        # ensures that min and max are integers
        min = int(user_min)
        max = int(user_max)

        if max < min:
            print('Error - max must be greater than min')
        else:
            if not os.path.exists('mults.csv'): # writes headers if file does not exist
                with open('mults.csv', 'w', newline='') as f:
                    csv.writer(f).writerow(['n', 'd1', 'd2', 'i', 'multiplicities'])

            for n in range(min, max+1): # max+1 ensures n=max runs
                main(n)
    except:
        print('Error - min and max must be integers')