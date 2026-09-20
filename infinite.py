from functions import get_primitive_int, generate_gf_tables, get_anf, get_AB_subfield, get_canonical_form, get_cubic_indices
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

def get_uniformity_multi(N, sbox, j, spaces_chunks, cores, count):
    tasks = [(N, chunk, sbox) for chunk in spaces_chunks]
    with ProcessPoolExecutor(max_workers=cores) as executor: # calculates if f(x) = xTr(ax^[2^j + 2^(2j)]) has optimal delta^2
        futures = [executor.submit(worker_chunk, t) for t in tasks] # submits batches of (a, b)-pairs to be processed
        with ProgressBar(total=count, desc='j = ' + str(j)) as pbar:
            for future in as_completed(futures): # as batches of functions are processed
                res, funcs = future.result()

                if res == 0: # if func is not optimal
                    return 0

                pbar.update(funcs)

    return 4

@njit(cache=True, nogil=True)
def get_uniformity(N, M, MOD, sbox, log_table, exp_table, pbar): # calculates if f(x) = x^d1 + a^i x^d2 has optimal delta^2
    # generates (a, b) pairs the same way as get_AB_subfield (see functions.py for the derivation),
    # checking the DDT immediately for each representative instead of storing it - avoids both the
    # O(N^2) visited array AND materializing the (huge, at n=22) pairs array
    dddt = np.zeros(N, dtype=np.int32)

    sub_size = 2**M - 1 # size of the subfield F_{2^M}*
    step = MOD // sub_size # = 2^M + 1, the number of F_{2^M}-lines

    # non-degenerate case: a is the element of {a,b,a^b} on the smallest-indexed line
    for j in range(step):
        a = exp_table[j]
        for b in range(1, N):
            if b == a:
                continue
            r_b = log_table[b] % step
            if r_b <= j:
                continue
            ab = a ^ b
            r_ab = log_table[ab] % step
            if r_ab <= j:
                continue
            if b < ab:
                for k in range(N):
                    dddt[k] = 0
                for x in range(N):
                    c = sbox[x] ^ sbox[x^a] ^ sbox[x^b] ^ sbox[x^ab]
                    dddt[c] += 1
                    if dddt[c] > 4:
                        return 0

                pbar.update()

    # degenerate case: a, b, a^b all on the same line - solved once for line 0, then mapped to every line
    EMB = np.zeros(sub_size, dtype=np.uint32)
    for k in range(sub_size):
        EMB[k] = exp_table[k*step]

    deg0_a = np.zeros(sub_size, dtype=np.uint32)
    deg0_b = np.zeros(sub_size, dtype=np.uint32)
    deg0_count = 0

    for i in range(sub_size):
        x = EMB[i]
        for jx in range(i+1, sub_size):
            y = EMB[jx]
            xy = x ^ y
            if y < xy: # canonical triple within line 0
                is_min = True # checks minimality under the (small, sub_size) full-line scaling orbit
                log_x = log_table[x]
                log_y = log_table[y]
                for k in range(1, sub_size):
                    shift = k*step
                    ux = exp_table[(log_x + shift) % MOD]
                    uy = exp_table[(log_y + shift) % MOD]
                    uxy = ux ^ uy

                    lo, mid, hi = ux, uy, uxy
                    if lo > mid:
                        lo, mid = mid, lo
                    if mid > hi:
                        mid, hi = hi, mid
                        if lo > mid:
                            lo, mid = mid, lo

                    if lo < x or (lo == x and mid < y):
                        is_min = False
                        break
                if is_min:
                    deg0_a[deg0_count] = x
                    deg0_b[deg0_count] = y
                    deg0_count += 1

    for j in range(step):
        a = exp_table[j]
        for i in range(deg0_count):
            x0 = deg0_a[i]
            y0 = deg0_b[i]
            if j == 0:
                lo, hi = x0, y0
            else:
                p = exp_table[(j + log_table[x0]) % MOD]
                q = exp_table[(j + log_table[y0]) % MOD]
                lo, hi = (p, q) if p < q else (q, p)
            ab = lo ^ hi

            for k in range(N):
                dddt[k] = 0
            for x in range(N):
                c = sbox[x] ^ sbox[x^lo] ^ sbox[x^hi] ^ sbox[x^ab]
                dddt[c] += 1
                if dddt[c] > 4:
                    return 0

            pbar.update()

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

        # formats output
        out_parts = []
        if zero_count > 0:
            out_parts.append(f"0:{zero_count}")
        for u, c in zip(unique_hits, counts_hits):
            out_parts.append(f"{u}:{c}")

        return '|'.join(out_parts)

def main(n, multi_processing):
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

    if multi_processing: # sets up multiprocessing chunks to max out CPU cores
        count, spaces = get_AB_subfield(N, M, exp_table, log_table, MOD)
        cores = multiprocessing.cpu_count()
        if n <= 13: # manages number of chunks to balance load
            num_chunks = 1
        elif n >= 14:
            num_chunks = 200
        elif n == 22:
            num_chunks = 10000

        spaces_chunks = np.array_split(spaces, num_chunks)
    else:
        count = int(((N-1)*(N-2)/6)/(2**M-1))

    for j in range(1, n):
        if math.gcd(j, n) == 1:
            tt = get_tt(N, M, MOD, j, exp_table, log_table) # gets truth table
            if multi_processing:
                delta2 = get_uniformity_multi(N, tt, j, spaces_chunks, cores, count)
            else:
                with ProgressBar(total=count, desc='j = ' + str(j)) as pbar:
                    delta2 = get_uniformity(N, M, MOD, tt, log_table, exp_table, pbar)

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
    multi_processing = config.get('infinite', 'multi_processing')
    
    try:
        # ensures that min and max are integers
        min2 = int(user_min)
        max2 = int(user_max)

        if max2 < min2:
            print('Error - max must be greater than min')
        elif multi_processing != 'True' and multi_processing != 'False':
            print('Error - mode must be True or False')
        else:
            mode = eval(multi_processing)
            for n in range(min2, max2+1): # max+1 ensures n=max runs
                if n in [6, 10, 14, 22]: # only works for n=2p where p is prime
                    print('n =', n)
                    main(n, mode)
                    print()
    except Exception as e:
        print(f'Error - min and max must be integers: {e}')