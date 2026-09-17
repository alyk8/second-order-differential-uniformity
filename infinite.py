from functions import get_primitive_int, generate_gf_tables, get_anf, get_AB, get_canonical_form, get_cubic_indices
from mults import calculate_mult
from configparser import ConfigParser # to read config file
from numba import njit # just-in-time compiler
import numpy as np
from numba_progress import ProgressBar
import math
import csv
import os # to check if csv files already exist
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_uniformtity(N, tt, j, spaces_chunks, cores, count):
    tasks = [(N, chunk, tt) for chunk in spaces_chunks]
    with ProcessPoolExecutor(max_workers=cores) as executor: # calculates if f(x) = xTr(ax^[2^j + 2^(2j)]) has optimal delta^2
        futures = [executor.submit(worker_chunk, t) for t in tasks] # submits batches of (a, b)-pairs to be processed
        with ProgressBar(total=count, desc='j = ' + str(j)) as pbar:
            for future in as_completed(futures): # as batches of functions are processed
                res, funcs = future.result()

                if res == 0: # if func is not optimal
                    return 0

                pbar.update(funcs)

    return 4

@njit(nogil=True)
def worker_chunk(args):
    N, spaces, sbox = args
    dddt = np.zeros(N, dtype=np.int32)

    for space in spaces:
        a = space[0]
        b = space[1]
        ab = space[2]

        for k in range(N):
            dddt[k] = 0
        for x in range(N):
            c = sbox[x] ^ sbox[x^a] ^ sbox[x^b] ^ sbox[x^ab]
            dddt[c] += 1
            if dddt[c] > 4:
                return 0, len(spaces)

    return 4, len(spaces)

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

    if not os.path.exists('optimal_infinites.csv'): # writes headers if file does not exist
        with open('optimal_infinites.csv', 'a', newline='') as f:
            csv.writer(f).writerow(['n', 'j', 'mult'])

    poly = get_primitive_int(n)
    exp_table, log_table = generate_gf_tables(N, MOD, poly)
    cubic_indices = get_cubic_indices(n, m)
    count, spaces = get_AB(N)

    # Setup multiprocessing chunks to max out CPU cores
    cores = multiprocessing.cpu_count()
    if n <= 13: # manages number of chunks to balance load
        num_chunks = 1
    elif n == 14:
        num_chunks = 200
    spaces_chunks = np.array_split(spaces, num_chunks)

    for j in range(1, n):
        if math.gcd(j, n) == 1:
            tt = get_tt(N, M, MOD, j, exp_table, log_table) # gets truth table
            delta2 = get_uniformtity(N, tt, j, spaces_chunks, cores, count)

            if delta2 == 4:
                mult = calculate_hist(m, tt, cubic_indices)
                with open('optimal_infinites.csv', 'a', newline='') as f:
                    csv.writer(f).writerow([n, j, mult])

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('infinite', 'min')
    user_max = config.get('infinite', 'max')
    
    try:
        # ensures that min and max are integers
        min2 = int(user_min)
        max2 = int(user_max)

        if max2 < min2:
            print('Error - max must be greater than min')
        else:
            for n in range(min2, max2+1): # max+1 ensures n=max runs
                if n in [6, 10, 14]: # only works for n=2p where p is prime
                    print('n =', n)
                    main(n)
                    print()
    except Exception as e:
        print(f'Error - min and max must be integers: {e}')