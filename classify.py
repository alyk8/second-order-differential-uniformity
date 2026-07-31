from functions import *
import numpy as np
from numba import njit
from configparser import ConfigParser # to read config file
import tqdm # for progress bar
import os
import csv

@njit
def calculate_invertibles(n): # gets all invertible matrices in GF(2^n)
    invertibles = np.zeros(shape=(31*30*28*24*16, n), dtype=np.uint8) # hardcoded matrix allocation size for n=5
    count = 0
    N = 2**n
    M = np.zeros(n, dtype=np.int64)
    
    for c1 in range(1, N):
        M[0] = c1
        for c2 in range(1, N):
            M[1] = c2
            if get_rank(n, 2, M[:2]) == 2:
                for c3 in range(1, N):
                    M[2] = c3
                    if get_rank(n, 3, M[:3]) == 3:
                        for c4 in range(1, N):
                            M[3] = c4
                            if get_rank(n, 4, M[:4]) == 4:
                                for c5 in range(1, N):
                                    M[4] = c5
                                    if get_rank(n, 5, M) == 5:
                                        invertibles[count, 0] = c1
                                        invertibles[count, 1] = c2
                                        invertibles[count, 2] = c3
                                        invertibles[count, 3] = c4
                                        invertibles[count, 4] = c5
                                        count += 1

    return count, invertibles

@njit
def get_cubic_coeffs(n, m): # gets all cubic indicies
    cubic_coeffs = np.zeros(m, dtype=np.uint64)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cubic_coeffs[count] = 2**i + 2**j + 2**k
                count += 1

    return cubic_coeffs

def get_functions(n, m): # gets all optimal functions in GF(2^n) into an array
    with open('search/optimal.csv', 'r') as f:
        lines = f.readlines()

    funcs = np.zeros(shape=(len(lines), m), dtype=np.uint64)
    count = 0 # no. of optimal functions
    for line in lines[1:]:
        data = line.strip().split(',') # gets one function without trailing spaces
        if int(data[0]) == n:
            data2 = data[1].strip().split('|')
            for j in range(m):
                funcs[count, j] = int(data2[j])
            count += 1

    return count, funcs[:count]

@njit
def pack_func(m, func): # converts a function to a 64-bit integer
    res = np.uint64(0)
    for i in range(m):
        res = (res << np.uint64(5) | np.uint64(func[i]))
    return res

@njit
def get_tt(n, m, cubic_coeffs, coeffs): # returns the 1D truth table of coeffs where each entry is a bit-packed integer
    N = 2**n
    tt1 = np.zeros(N, dtype=np.uint64)
    for x in range(N):
        for c in range(m): # cubic coefficients only
            if (x & cubic_coeffs[c]) == cubic_coeffs[c]:
                tt1[x] ^= coeffs[c]

    return tt1

@njit
def get_g(n, m, cubic_coeffs, inv, tt1, funcs_packed, active_mask, high): # multiplies a function from its truth table by an invertible matrix
    N = 2**n
    tt2 = np.zeros(N, dtype=np.uint64) # truth table of L_2 (i.e. inv)

    # multiplies the input vector 'x' by the matrix 'inv'
    for x in range(N): # for each input i
        val = 0
        for row_idx in range(n):
            if (x >> row_idx) & 1: # if the row_idx-th bit of x is 1
                val ^= inv[row_idx]
        tt2[x] = val

    # re-maps the truth table
    anf = np.zeros(N, dtype=np.uint64) # truth table of F*L_2
    for i in range(N):
        anf[i] = tt1[tt2[i]]

    # butterfly algorithm to convert the truth table into ANF
    for i in range(n):
        step = 2**i
        for j in range(0, N, step*2):
            for k in range(j, j+step):
                anf[k+step] ^= anf[k]

    # extracts coeffs and maps to canonical form
    coeffs = np.zeros(m, dtype=np.uint64)
    for i in range(m):
        coeffs[i] = anf[cubic_coeffs[i]]

    g_prime = get_canonical_form(n, m, coeffs)
    g_prime_packed = pack_func(m, g_prime)

    low = 0
    while low <= high:
        mid = (low + high)//2
        if funcs_packed[mid] == g_prime_packed:
            if active_mask[mid]:
                active_mask[mid] = False
                return 1
            else:
                return 0
        elif funcs_packed[mid] < g_prime_packed:
            low = mid + 1
        else:
            high = mid - 1

    return 0

def main(n):
    m = int((n*(n-1)*(n-2))/6) # number of coeffs, i.e. nC3
    
    count_inv, invertibles = calculate_invertibles(n)
    cubic_coeffs = get_cubic_coeffs(n, m)

    total_func, funcs = get_functions(n, m)
    funcs_packed = np.array([pack_func(m, f) for f in funcs], dtype=np.uint64) # converts all functions into integers

    sort_idx = np.argsort(funcs_packed)
    funcs_packed = funcs_packed[sort_idx]
    funcs = funcs[sort_idx]
    active_mask = np.ones(total_func, dtype=np.bool_) # create a boolean mask to track which functions are still active

    count_func = 0
    while count_func < total_func:
        for idx in range(total_func):
            if active_mask[idx]:
                func = funcs[idx]
                break

        tt1 = get_tt(n, m, cubic_coeffs, func) # gets truth table of coeffs

        print('\nChecking function', func)
        with tqdm.tqdm(total=total_func, initial=count_func, desc='Remaining Functions') as pbarFunc:
            with tqdm.tqdm(total=count_inv, desc='Applying invertibles') as pbarInv:
                for k in range(count_inv):
                    num = get_g(n, m, cubic_coeffs, invertibles[k], tt1, funcs_packed, active_mask, total_func-1)
                    if num == 1:
                        count_func += 1
                        pbarFunc.update()
                    pbarInv.update()

        if not os.path.exists('optimal' + str(n) + '.csv'): # writes headers if file does not exist
            with open('optimal' + str(n) + '.csv', 'a', newline='') as f:
                csv.writer(f).writerow(['n', 'coeffs'])
            
        with open('optimal' + str(n) + '.csv', 'a', newline='') as f:
            func_string = '|'.join(map(str, func))
            csv.writer(f).writerow([n, func_string])
    
if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')
    n = config.get('classify', 'n')

    try:
        n = int(n) # ensures that n is an integer
        main(n)
    except ValueError:
        print('Error - n must be an integer')