import numpy as np
from numba import njit
from configparser import ConfigParser # to read config file
import tqdm # for progress bar
import os
import csv

@njit
def int_to_vec(n, i): # converts an integer to its vector form
    v = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        v[j] = (i >> j) & 1 # right shift to get jth bit
    return v

@njit
def getRank(A): # returns the rank of a matrix M
    n = A.shape[0] # the no. of rows and columns
    rank = 0

    for c in range(n): # for each column c
        pivot = -1
        for r in range(rank, n): # finds a row r that has a 1 in column c
            if A[r, c] == 1:
                pivot = r # sets the pivot to the row
                break
        
        if pivot != -1: # if column c is not all 0s
            if pivot != rank: # swaps pivot row into rank row
                for col in range(n):
                    temp = A[rank, col]
                    A[rank, col] = A[pivot, col]
                    A[pivot, col] = temp

            for r in range(n): # eliminates column c from all non-pivot rows
                if r != rank and A[r, c] == 1:
                    for col in range(n):
                        A[r, col] ^= A[rank, col] # XORs row with the pivot row
            
            rank += 1
            if rank == n: # rank can't be more than the no. of rows
                return rank

    return rank

@njit
def getAB(n): # returns all (a, a+b) pairs to be checked
    size = int((2**n-1)*(2**n-2)/6) # the no. of pairs
    diffs = np.zeros(shape=(size, 2, n), dtype=np.uint8) # stores (a, a+b) pairs in vector form
    count = 0

    for a in range(1, 2**n): # excludes the zero vector
        seen = np.zeros(2**n, dtype=np.uint8) # records all of the b's for this specific a
        for ab in range(a+1, 2**n): # a < a^b
            if not seen[ab]: # if we haven't already set b to this a^b
                b = a^ab # a^(a^b) = (a^a)^b = 0^b = b
                seen[b] = 1
                if ab < b: # if we haven't already set a to this a^b
                    diffs[count, 0] = int_to_vec(n, a)
                    diffs[count, 1] = int_to_vec(n, ab)
                    count += 1
    
    return diffs

@njit
def isOpt(n, m, func, diffs): # checks if function has second-order differential uniformity 4
    coeffs = np.zeros(shape = (m, n), dtype=np.uint8) # coeffs = [[coeffs of x1x2x3], ..., [coeffs of x(n-2)x(n-1)xn]]
    for i in range(m): # for each cubic coefficient
        coeffs[i] = int_to_vec(n, func[i]) # adds coefficient in vector form

    v = np.zeros(shape=(n, n), dtype=np.uint8) # allocates one time
    num = diffs.shape[0]
    for d in range(num): # for each (a, b) pair
        a = diffs[d, 0]
        b = diffs[d, 1]

        for i in range(n):
            for j in range(n):
                v[i, j] = 0

        c = 0 # coefficient no.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n): # i < j < k
                    if (a[j] & b[k])^(a[k] & b[j]):
                        for col in range(n):
                            v[i, col] ^= coeffs[c, col]
                    if (a[i] & b[k])^(a[k] & b[i]):
                        for col in range(n):
                            v[j, col] ^= coeffs[c, col]
                    if (a[i] & b[j])^(a[j] & b[i]):
                        for col in range(n):
                            v[k, col] ^= coeffs[c, col]
                    c += 1
        
        rank = getRank(v)
        if rank < n-2: # function is not optimal
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
def getFunctions(n, m): # gets all canonical cubic functions in a field
    rank = n - 2 # min rank to be optimal
    max = 2**((n - 1)*(m - rank)) # max no. of canonical functions
    initialCoeffs = np.zeros(m, dtype=np.uint8)
    results = np.zeros(shape=(max, m), dtype=np.uint8)
    count = np.zeros(1, dtype=np.uint64)

    for i in range(rank): # sets the first n-2 coeffs to e vectors
        initialCoeffs[i] = 2**i

    canonical(n, m, rank, initialCoeffs, rank, results, count)

    return count[0], results[:count[0]].copy()

def main(n):
    m = int((n*(n-1)*(n-2))/6) # number of coeffs, i.e. nC3
    count, funcs = getFunctions(n, m) # gets all canonical functions in that field
    diffs = getAB(n) # gets all (a, b) pairs to check

    with tqdm.tqdm(total = count, desc='n = ' + str(n)) as pbar:
        for func in funcs:
            delta = isOpt(n, m, func, diffs)
            if delta == 4: # if function is optimal
                func_string = '|'.join(map(str, func))
                with open('optimal.csv', 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([n, func_string])
            pbar.update() # update progress bar
    print()

if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')

    user_min = config.get('search', 'min')
    user_max = config.get('search', 'max')

    try:
        # ensures that min and max are integers
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
    except:
        print('Error - min and max must be integers')