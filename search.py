from functions import *
import numpy as np
from numba import njit
from configparser import ConfigParser # to read config file
import tqdm # for progress bar
import os
import csv

@njit
def get_ab(N): # returns all (a, a+b) pairs to be checked
    size = int((N-1)*(N-2)/6) # the no. of pairs
    diffs = np.zeros(shape=(size, 2), dtype=np.int64) # stores (a, a+b) pairs as packed integers
    count = 0

    for a in range(1, N): # excludes the zero vector
        seen = np.zeros(N, dtype=np.uint8) # records all of the b's for this specific a
        for ab in range(a+1, N): # a < a^b
            if not seen[ab]: # if we haven't already set b to this a^b
                b = a^ab # a^(a^b) = (a^a)^b = 0^b = b
                seen[b] = 1
                if ab < b: # if we haven't already set a to this a^b
                    diffs[count, 0] = a
                    diffs[count, 1] = ab
                    count += 1
    
    return diffs

@njit
def is_opt(n, func, diffs): # checks if function has second-order differential uniformity 4
    M = np.zeros(n, dtype=np.int64) # allocates one time

    for d in range(diffs.shape[0]): # for each (a, b) pair
        build_M(n, diffs[d, 0], diffs[d, 1], M, func) # builds the matrix M for the 2D subspace
        
        if get_rank(n, n, M) < n-2: # function is not optimal
            return 0
    
    return 4

@njit
def canonical(n, m, depth, coeffs, rank, results, count): # recursive function for a depth-first search
    if depth == m: # if all the coeffs have been assigned
        results[count[0]] = coeffs # adds function to array
        count[0] += 1
        return # exits function call

    for i in range(2**rank): # adds all the vectors in the current span as the next coeff
        new_coeffs = coeffs.copy()
        new_coeffs[depth] = i # changes the next coeff to every number from 0 to 2^rank-1 (this is the span)
        canonical(n, m, depth+1, new_coeffs, rank, results, count) # moves to the next coeff

    if rank < n: # increases the rank
        new_coeffs = coeffs.copy()
        new_coeffs[depth] = 2**rank # sets next coefficient to the next e vector
        canonical(n, m, depth+1, new_coeffs, rank+1, results, count) # moves to next rank

@njit
def get_functions(n, m): # gets all canonical cubic functions in a field
    rank = n-2 # min rank to be optimal
    max = 2**((n-1)*(m-rank)) # max no. of canonical functions
    initial_coeffs = np.zeros(m, dtype=np.int64)
    results = np.zeros(shape=(max, m), dtype=np.int64)
    count = np.zeros(1, dtype=np.uint64)

    for i in range(rank): # sets the first n-2 coeffs to e vectors
        initial_coeffs[i] = 2**i

    canonical(n, m, rank, initial_coeffs, rank, results, count)

    return count[0], results[:count[0]].copy()

def main(n):
    N = 2**n
    m = int((n*(n-1)*(n-2))/6) # number of coeffs, i.e. nC3

    count, funcs = get_functions(n, m) # gets all canonical functions in that field
    diffs = get_ab(N) # gets all (a, b) pairs to check

    with tqdm.tqdm(total=count, desc='n = ' + str(n)) as pbar:
        for func in funcs:
            delta = is_opt(n, func, diffs)
            if delta == 4: # if function is optimal
                func_string = '|'.join(map(str, func))
                with open('optimal.csv', 'a+', newline='') as f:
                    csv.writer(f).writerow([n, func_string])
            pbar.update() # update progress bar
    print()

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('search', 'min')
    user_max = config.get('search', 'max')

    try: # ensures that min and max are integers
        min = int(user_min)
        max = int(user_max)

        if max < min:
            print('Error - max must be greater than min')
        else:
            print()
            if not os.path.exists('optimal.csv'): # writes headers if file does not exist
                with open('optimal.csv', 'a', newline='') as f:
                    csv.writer(f).writerow(['n', 'coeffs'])
        
            for n in range(min, max+1): # fields to check
                main(n)
    except Exception as e:
            print(f'Error - min and max must be integers: {e}')