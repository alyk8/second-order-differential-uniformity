from functions import *
import numpy as np
from numba import njit, prange, types
from numba.typed import Dict
from numba_progress import ProgressBar
from configparser import ConfigParser # to read config file
import time

@njit
def build_space_lookup(count, spaces): # creates a fast hash map linking a 2D subspace to its array index
    lookup = Dict.empty(key_type=types.uint64, value_type=types.int32) # packs the 3 non-zero elements of the subspace packed into a single integer key
    for i in range(count):
        a = np.uint64(spaces[i, 0])
        b = np.uint64(spaces[i, 1])
        c = np.uint64(spaces[i, 2])
        key = a | (b << np.uint64(20)) | (c << np.uint64(40))
        lookup[key] = np.int32(i)

    return lookup

@njit()
def get_orthoderivative(n, coeffs, count, spaces, lookup): # computes the orthoderivative mapping for a boolean function
    pi_map = np.full(count, -1, dtype=np.int32)
    
    for i in prange(count):
        U = spaces[i]
        a = U[0]
        b = U[1]

        # builds the coefficient matrix M for the second-order derivative
        M = np.zeros(n, dtype=np.uint32)
        c = 0
        for x in range(n):
            ax = (a >> x) & 1
            bx = (b >> x) & 1
            for y in range(x + 1, n):
                ay = (a >> y) & 1
                by = (b >> y) & 1
                for z in range(y + 1, n):
                    az = (a >> z) & 1
                    bz = (b >> z) & 1
                    
                    term_a = (ay & bz) ^ (az & by)
                    term_b = (ax & bz) ^ (az & bx)
                    term_c = (ax & by) ^ (ay & bx)
                    
                    if term_a: M[x] ^= coeffs[c]
                    if term_b: M[y] ^= coeffs[c]
                    if term_c: M[z] ^= coeffs[c]
                    c += 1

        # gets the kernel of M and maps it to a valid subspace index using the lookup
        K = get_kernel_basis(n, M)
        if K[1] != 0:
            key = np.uint64(K[1]) | (np.uint64(K[2]) << np.uint64(20)) | (np.uint64(K[3]) << np.uint64(40))
            if key in lookup:
                pi_map[i] = lookup[key]

    return pi_map

@njit(inline='always')
def are_adjacent(U, V): # determines if U and V are adjacent in the Grassmann graph, i.e. they share exactly one non-zero element
    return (U[0] == V[0] or U[0] == V[1] or U[0] == V[2] or
            U[1] == V[0] or U[1] == V[1] or U[1] == V[2] or
            U[2] == V[0] or U[2] == V[1] or U[2] == V[2])

@njit(parallel=True, nogil=True)
def check_automorphism(N, pi_map, count, spaces, lookup, progress): # calculates the total mapping violations and per-vertex violation profiles
    per_vertex = np.zeros(count, dtype=np.int32)
    pi_counts = np.zeros(count, dtype=np.int32)

    # pre-count the image space collisions
    N_bad = 0
    for i in range(count):
        p = pi_map[i]
        if p != -1:
            pi_counts[p] += 1
        else:
            N_bad += 1
            
    for i in prange(count):
        pi_i = pi_map[i]
        if pi_i == -1:
            per_vertex[i] = count - 1
            continue

        U = spaces[i]
        count_A_valid = 0
        k1 = 0
        
        # evaluates graph neighbours
        for idx in range(0, 3):
            x = U[idx]
            for y in range(1, N):
                if y == x:
                    continue
                z = x^y
                if y < z:
                    a, b, c = x, y, z
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    key = np.uint64(a) | (np.uint64(b) << np.uint64(20)) | (np.uint64(c) << np.uint64(40))
                    
                    if key in lookup:
                        j = lookup[key]
                        if j != i:
                            pi_j = pi_map[j]
                            if pi_j != -1:
                                count_A_valid += 1
                                if are_adjacent(spaces[pi_j], spaces[pi_i]):
                                    k1 += 1
                                
        # evaluates mapped graphic neighbours
        sum_B = 0
        W_img = spaces[pi_i]
        for idx in range(0, 3):
            x = W_img[idx]
            for y in range(1, N):
                if y == x: continue
                z = x ^ y
                if y < z:
                    a, b, c = x, y, z
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    key = np.uint64(a) | (np.uint64(b) << np.uint64(20)) | (np.uint64(c) << np.uint64(40))
                    
                    if key in lookup:
                        W_j = lookup[key]
                        if W_j != pi_i:
                            sum_B += pi_counts[W_j]

        # sums total violations using internal mapping collisions                    
        per_vertex[i] = N_bad + count_A_valid - 2*k1 + sum_B + (pi_counts[pi_i] - 1)

        progress.update(1) # updates progress bar
        
    return per_vertex

def check_automorphisms(N, pi_map, count, spaces, lookup, func_str): # initialises the progress bar
    with ProgressBar(total=count, desc=func_str) as progress:
        profile = check_automorphism(N, pi_map, count, spaces, lookup, progress)
    total_violations = np.sum(np.int64(profile)) // 2 # the total network violations over all pairs is the sum of profile // 2

    return total_violations, profile

def build_field_context(n, N, MOD): # initialises static data structures for a field n
    m = int((n*(n-1)*(n-2))/6) # no. of monomials
    poly = get_primitive_int(n)
    exp_t, log_t = generate_gf_tables(N, MOD, poly)

    # gets cubic coefficients of the ANF
    cubic_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cubic_indices.append((1<<i) | (1<<j) | (1<<k))
    cubic_indices = np.array(cubic_indices, dtype=np.uint32)
    
    count, subspaces = get_AB(N, False) # gets all ab-pairs, including pairs in the same class
    lookup = build_space_lookup(count, subspaces)
    
    return m, exp_t, log_t, cubic_indices, count, subspaces, lookup

def test_polynomial(n, N, d1, d2, k, ctx, file_obj, MOD): # calculates and saves the profiles of the function f(x) = x^d1 + a^i x^d2
    m, exp_t, log_t, cubic_indices, count, subspaces, lookup = ctx
    tt = get_tt(N, d1, d2, k, exp_t, log_t, MOD) # gets truth table
    anf = get_anf(n, tt) # gets the ANF of the truth table

    # gets cubic coefficients of the ANF
    coeffs = np.zeros(m, dtype=np.uint32)
    for i in range(m):
        coeffs[i] = anf[cubic_indices[i]]

    # formats, prints and saves output in textfile
    if coeffs.any():
        func_str = f'x^{d1}' if d2 == -1 else f'x^{d1} + α^{k}·x^{d2}'
        pi_map = get_orthoderivative(n, coeffs, count, subspaces, lookup)
        n_violations, profile = check_automorphisms(N, pi_map, count, subspaces, lookup, func_str)
            
        print(f'{n_violations} violations\n', end='')
        file_obj.write(func_str + f': {n_violations} violations\n')
        
        unique, counts = np.unique(profile, return_counts=True)
        for val, cnt in zip(unique, counts):
            line_str = f"  {cnt} vertices each violate for {val} neighbours\n"
            print(line_str, end='')
            file_obj.write(line_str)
            
        print()
        file_obj.write('\n')
        file_obj.flush()

def main(n, f, min):
    start = time.time()
    n_header = f'n = {n}\n'
    if n > min:
        n_header = '\n' + n_header
        
    print(n_header, end='')
    f.write(n_header)

    N = 2**n
    MOD = N - 1
    ctx = build_field_context(n, N, MOD)
    exps = get_exponents(n, MOD)
    
    for case in exps:
        d1, d2, k = case[0], case[1], case[2]
        test_polynomial(n, N, d1, d2, k, ctx, f, MOD)

    end = time.time()
    print(f"Time taken: {round(end-start, 2)} seconds")

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('grassmann', 'min')
    user_max = config.get('grassmann', 'max')

    try:
        # ensures that min and max are integers
        min = int(user_min)
        max = int(user_max)

        if max < min:
            print('Error - max must be greater than min')
        else:
            with open('grassmann.txt', 'a', encoding='utf-8') as f: # utf-8 ensures alpha can be written to the textfile
                for n in range(min, max+1):
                    main(n, f, min)
    except Exception as e:
        print(f'Error - min and max must be integers: {e}')