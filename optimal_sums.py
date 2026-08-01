from functions import *
from numba import njit # just-in-time compiler
from configparser import ConfigParser # to read config file
import numpy as np
import tqdm # for progress bar
import csv
import os # to check if csv files already exist
import math

def get_new_ivalues(r, n): # gets possible i values for a given r = gcd(d_2-d_1, 2^n-1) and n
    if r <= 1:
        return [0]
    seen = np.zeros(r, dtype=np.uint8)
    reps = []
    for i in range(r):
        if seen[i] == 0:
            reps.append(i)
            temp = i
            for _ in range(n):
                seen[temp] = 1
                temp = (temp*2) % r
    
    return reps

@njit
def get_uniformity(N, MOD, d1, d2, i, exp_table, log_table, spaces): # calculates delta^2 of f(x) = x^d1 + a^i x^d2
    sbox = np.empty(N, dtype=np.int32)
    sbox[0] = 0
    for x in range(1, N):
        idx = log_table[x]
        sbox[x] = exp_table[(idx*d1) % MOD] ^ exp_table[(idx*d2 + i) % MOD]

    delta2 = np.int32(0)
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
        current_max = np.max(dddt)
        if current_max > delta2:
            delta2 = current_max

    return delta2

@njit
def get_uniformity_part(N, MOD, d1, d2, i, exp_table, log_table, spaces): # calculates if f(x) = x^d1 + a^i x^d2 has optimal delta^2
    sbox = np.empty(N, dtype=np.int32)
    sbox[0] = 0
    for x in range(1, N):
        idx = log_table[x]
        sbox[x] = exp_table[(idx*d1) % MOD] ^ exp_table[(idx*d2 + i) % MOD]

    dddt = np.zeros(N, dtype=np.int32)
    for space in spaces:
        for k in range(N):
            dddt[k] = 0
        for x in range(N):
            c = sbox[x] ^ sbox[x^space[0]] ^ sbox[x^space[1]] ^ sbox[x^space[2]]
            dddt[c] += 1
            if dddt[c] > 4:
                return 0
    
    return 4

@njit
def build_pairs(exps_d, exps_rot, orbit_ids, MOD, r_counts): # gets all valid (d1, d2) pairs to test
    no_of_exps = len(exps_d)

    # PASS 1: counts valid pairs so we can set size of array
    pair_count = 0 # no. of (d1, d2) pairs to test
    i_count = np.int64(0) # no. of (d1, d2, i) sets to test
    for k in range(no_of_exps):
        if exps_rot[k] == 0: # forces d1 to be the base element of its cyclotomic coset
            d1 = exps_d[k]
            for j in range(k+1, no_of_exps): # ensures d2 > d1
                d2 = exps_d[j]
                
                if orbit_ids[d2-1] != d1: # ensures d1 and d2 do not belong to the same cyclotomic coset
                    r = math.gcd(int(d2) - int(d1), int(MOD)) # the GCD determines how many i values to test
                    i_count += r_counts[r]
                    pair_count += 1

    # PASS 2: allocates and fills the array
    ds = np.empty((pair_count, 2), dtype=np.int32)
    idx = 0
    for k in range(no_of_exps):
        if exps_rot[k] == 0:
            d1 = exps_d[k]
            for j in range(k+1, no_of_exps):
                d2 = exps_d[j]
                
                if orbit_ids[d2-1] != d1:
                    ds[idx, 0] = d1
                    ds[idx, 1] = d2
                    idx += 1
    
    return ds, i_count

def get_exponents(n, N, MOD): # gets all unique exponent values for d_1 and d_2
    seen = np.zeros(MOD, dtype=np.bool_)
    orbit_ids = np.zeros(MOD, dtype=np.int32) # maps an exponent to the canonical rep of its orbit
    exps_d_list = []
    exps_rot_list = []
    half_n = n // 2 # caps rotations at n/2 to prevent testing symmetric binomials twice

    for d in range(1, N): # for each possible exponent
        if not seen[d-1]: # if it's from a new cyclotomic class
            seen[d-1] = True
            
            degree = bin(d).count('1') # the binary weight of 'd' is equal to the algebraic degree of x^d
            orbit_ids[d-1] = d
            
            if degree > 2: # removes degree 1 (affine) and 2 (quadratic) functions
                exps_d_list.append(d)
                exps_rot_list.append(0) # 0 rotation means this is the base canonical element
            
            temp = d*2 % MOD # generates the rest of its cyclotomic class by multiplying by 2 (bit-shifting)
            rotation = 1
            while not seen[temp-1]:
                seen[temp-1] = True
                orbit_ids[temp-1] = d
                
                if degree > 2 and rotation <= half_n: # only keep elements with degree > 2 and within our symmetry limits
                    exps_d_list.append(temp)
                    exps_rot_list.append(rotation)
                
                temp = temp*2 % MOD
                rotation += 1

    # converts lists to NumPy arrays and sorts them by exponent value
    exps_d = np.array(exps_d_list, dtype=np.int32)
    exps_rot = np.array(exps_rot_list, dtype=np.int32)
    order = np.argsort(exps_d)
    exps_d = exps_d[order]
    exps_rot = exps_rot[order]
    
    # precomputes how many valid alpha^i scalars exist for each r = gcd(d_2-d_1, 2^n-1) of MOD
    r_counts = np.zeros(N, dtype=np.int64)
    for r in range(1, N):
        if MOD % r == 0:
            r_counts[r] = len(get_new_ivalues(r, n))
    
    return build_pairs(exps_d, exps_rot, orbit_ids, MOD, r_counts)

def main(n, mode):
    N = 2**n
    MOD = N-1

    poly = get_primitive_int(n)
    exp_table, log_table = generate_gf_tables(N, MOD, poly)

    ds, count = get_exponents(n, N, MOD)
    _, spaces = get_AB(N) # unique (a, b) pairs

    if not os.path.exists('optimal_sums.csv'): # writes headers if file does not exist
        with open('optimal_sums.csv', 'a', newline='') as f:
            csv.writer(f).writerow(['n', 'd1', 'd2', 'i', 'delta2'])
    
    if mode == 'A':
        if not os.path.exists(str(n) + ' (sums).csv'):
            with open(str(n) + ' (sums).csv', 'a', newline='') as f:
                csv.writer(f).writerow(['d1', 'd2', 'i', 'delta2'])
    
    with tqdm.tqdm(total=count, desc='n = ' + str(n)) as pbar:
        for d in ds:
            d1, d2 = d[0], d[1]
            r = math.gcd(d2-d1, MOD)
            iValues = get_new_ivalues(r, n)

            for i in iValues:
                if mode == 'A': # all mode
                    delta2 = get_uniformity(N, MOD, d1, d2, i, exp_table, log_table, spaces)
                    with open(str(n) + ' (sums).csv', 'a', newline='') as f:
                        csv.writer(f).writerow([d1, d2, i, delta2])
                else: # part mode, i.e. optimals only
                    delta2 = get_uniformity_part(N, MOD, d1, d2, i, exp_table, log_table, spaces)
                
                if delta2 == 4: # if function is optimal, saves in textfile
                    with open('optimal_sums.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([n, d1, d2, i, delta2])
                pbar.update()

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('sums', 'min')
    user_max = config.get('sums', 'max')
    mode = config.get('sums', 'mode') # A (all) or P (part)

    try:
        # ensures that min and max are integers
        min = int(user_min)
        max = int(user_max)

        if max < min:
            print('Error - max must be greater than min')
        elif mode != 'A' and mode != 'P':
            print('Error - mode must be A (all) or P (part)')
        else:
            for n in range(min, max+1): # max+1 ensures n=max runs
                print()
                main(n, mode)
    except Exception as e:
        print(f'Error - min and max must be integers: {e}')