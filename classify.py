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
def vec_to_int(n, v): # converts a vector to its respective integer
    i = 0
    for j in range(n):
        i += v[j]*(2**j)
    return i

@njit
def getRank(M): # returns the rank of a matrix M
    A = M.copy() # does not mainpulate original matrix
    n, m = M.shape # the no. of rows and columns
    rank = 0

    for c in range(m): # for each column c
        pivot = -1
        for r in range(rank, n): # finds a row r that has a 1 in column c
            if A[r, c] == 1:
                pivot = r # sets the pivot to the row
                break
        
        if pivot != -1: # if column c is not all 0s
            if pivot != rank: # swaps pivot row into rank row
                temp = A[rank].copy()
                A[rank] = A[pivot]
                A[pivot] = temp

            for r in range(n): # eliminates column c from all non-pivot rows
                if r != rank and A[r, c] == 1:
                    A[r] = A[r]^A[rank] # XORs row with the pivot row
            
            rank += 1
            if rank == n: # rank can't be more than the no. of rows
                return rank

    return rank

@njit
def getInverse(M): # returns the inverse of a square matrix M
    A = M.copy() # does not mainpulate original matrix
    n = M.shape[0] # the no. of rows and columns
    I = np.eye(n, dtype=np.uint8)
    rank = 0

    for c in range(n): # for each column c
        pivot = -1
        for r in range(rank, n): # finds a row r that has a 1 in column c
            if A[r, c] == 1:
                pivot = r # sets the pivot to the row
                break
        
        if pivot != -1: # if column c is not all 0s
            if pivot != rank: # swaps pivot row into rank row
                temp = A[rank].copy()
                A[rank] = A[pivot]
                A[pivot] = temp

                temp = I[rank].copy()
                I[rank] = I[pivot]
                I[pivot] = temp

            for r in range(n): # eliminates column c from all non-pivot rows
                if r != rank and A[r, c] == 1:
                    A[r] = A[r]^A[rank] # XORs row with the pivot row
                    I[r] = I[r]^I[rank]

            rank += 1

    if rank < n:
        return np.zeros(shape=(n, n), dtype=np.uint8)
    
    return I

@njit
def getCanconicalForm(n, m, coeffs):
    basis = np.zeros(shape = (n, n), dtype=np.uint8)
    count = 0
    for i in range(m):
        basis[count] = int_to_vec(n, coeffs[i])
        if getRank(basis[:count+1]) == count+1:
            count += 1
            if count == n:
                break

    L1 = getInverse(basis.transpose())

    G = np.zeros(shape = (m, n), dtype=np.uint8)
    for i in range(m):
        G[i] = int_to_vec(n, coeffs[i])

    G2 = np.zeros(shape = (m, n), dtype=np.uint8)
    for j in range(m):
        for i in range(n):
            for k in range(n):
                G2[j, i] ^= L1[i, k] & G[j, k] # row * row (transposed column)

    gPrime = np.zeros(m, dtype=np.uint8)
    for i in range(m):
        gPrime[i] = vec_to_int(n, G2[i])

    return gPrime

@njit
def calculateInvertibles(n): # gets all invertible matrices in GF(2^n)
    invertibles = np.zeros(shape=(31*30*28*24*16, n, n), dtype=np.uint8)
    vectors = np.zeros(shape = (2**n-1, n), dtype=np.uint8) # generates all vectors in GF(2^n)
    for i in range(1, 2**n): # only do this computation once per n
        vectors[i-1] = int_to_vec(n, i)
    
    count = 0
    for c1 in vectors:
        M = np.zeros(shape=(n, n), dtype=np.uint8)
        M[0] = c1
        for c2 in vectors:
            M[1] = c2
            if getRank(M[:2]) == 2:
                for c3 in vectors:
                    M[2] = c3
                    if getRank(M[:3]) == 3:
                        for c4 in vectors:
                            M[3] = c4
                            if getRank(M[:4]) == 4:
                                for c5 in vectors:
                                    M[4] = c5
                                    if getRank(M) == 5:
                                        invertibles[count] = M
                                        count += 1

    return count, invertibles

@njit
def getCubicCoeffs(n, m): # gets all cubic indicies
    cubicCoeffs = np.zeros(m, dtype=np.uint8)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cubicCoeffs[count] = 2**i + 2**j + 2**k
                count += 1
    return cubicCoeffs

def getFunctions(n, m): # gets all optimal functions in GF(2^n) into an array
    with open('search/optimal.csv', 'r') as f:
        lines = f.readlines()

    funcs = np.zeros(shape=(len(lines), m), dtype=np.uint8)
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
def packFunc(m, func): # converts a function to a 64-bit integer
    res = np.uint64(0)
    for i in range(m):
        res = (res << np.uint64(5) | np.uint64(func[i]))
    return res

@njit
def unpackFunc(m, packed): # converts an integer back to its coefficients
    func = np.zeros(m, dtype=np.uint8)
    for i in range(m - 1, -1, -1):
        func[i] = np.uint8(packed & np.uint64(0x1F))
        packed >>= np.uint64(5)
    return func

@njit
def getTT(n, m, cubicCoeffs, coeffs): # returns the truth table of coeffs
    tt1 = np.zeros(shape = (2**n, n), dtype=np.uint8)
    for x in range(2**n):
        for c in range(m): # cubic coefficients only
            if x & cubicCoeffs[c] == cubicCoeffs[c]:
                tt1[x] ^= coeffs[c]
    return tt1

@njit
def getG(n, m, cubicCoeffs, inv, vectors, tt1, funcsPacked, activeMask, high): # multiplies a function from its truth table by an invertible matrix
    tt2 = np.zeros(shape = (2**n, n), dtype=np.uint8) # truth table of L_2 (i.e. inv)
    for i in range(2**n): # for each input i
        for j in range(n): # for each col in inv
            for k in range(n): # for each row in inv
                tt2[i, j] ^= vectors[i, k] & inv[k, j] # multiply instead of XOR

    anf = np.zeros(shape = (2**n, n), dtype=np.uint8) # truth table of F*L_2
    for i in range(2**n):
        anf[i] = tt1[vec_to_int(n, tt2[i])]

    for i in range(0, n): # butterfly algorithm
        step = 2**i
        for j in range(0, 2**n, step*2):
            anf[j+step : j+2*step, :] ^= anf[j : j+step, :]

    coeffs = np.zeros(m, dtype=np.uint8)
    for i in range(m):
        coeffs[i] = vec_to_int(n, anf[cubicCoeffs[i]])

    gPrime = getCanconicalForm(n, m, coeffs)
    gPrimePacked = packFunc(m, gPrime)

    low = 0
    while low <= high:
        mid = (low + high)//2
        if funcsPacked[mid] == gPrimePacked:
            if activeMask[mid]:
                activeMask[mid] = False
                return 1
            else:
                return 0
        elif funcsPacked[mid] < gPrimePacked:
            low = mid + 1
        else:
            high = mid - 1

    return 0

def main(n):
    m = int((n*(n-1)*(n-2))/6) # number of coeffs, i.e. nC3
    
    countInv, invertibles = calculateInvertibles(n)
    cubicCoeffs = getCubicCoeffs(n, m)

    vectors = np.zeros(shape = (2**n, n), dtype=np.uint8) # generates all vectors in GF(2^n)
    for i in range(2**n): # only do this computation once per n
        vectors[i] = int_to_vec(n, i)

    totalFunc, funcs = getFunctions(n, m)
    funcsPacked = np.array([packFunc(m, f) for f in funcs], dtype=np.uint64) # converts all functions into integers
    sort_idx = np.argsort(funcsPacked)
    funcsPacked = funcsPacked[sort_idx]
    funcs = funcs[sort_idx]
    activeMask = np.ones(totalFunc, dtype=np.bool_) # create a boolean mask to track which functions are still active

    countFunc = 0
    while countFunc < totalFunc:
        for idx in range(totalFunc):
            if activeMask[idx]:
                func = funcs[idx]
                break

        coeffs = np.zeros(shape = (m, n), dtype=np.uint8) # coeffs = [[coeffs of x1x2x3], ..., [coeffs of x(n-2)x(n-1)xn]]
        for j in range(m): # for each cubic coefficient
            coeffs[j] = int_to_vec(n, func[j]) # adds coefficient in vector form
        tt1 = getTT(n, m, cubicCoeffs, coeffs) # gets truth table of coeffs

        print('\nChecking function', func)
        with tqdm.tqdm(total=totalFunc, initial=countFunc, desc='Remaining Functions') as pbarFunc:
            with tqdm.tqdm(total = countInv, desc='Applying invertibles') as pbarInv:
                for k in range(countInv):
                    num = getG(n, m, cubicCoeffs, invertibles[k], vectors, tt1, funcsPacked, activeMask, totalFunc-1)
                    if num == 1:
                        countFunc += 1
                        pbarFunc.update()
                    pbarInv.update()

        if not os.path.exists('optimal' + str(n) + '.csv'): # writes headers if file does not exist
            with open('optimal' + str(n) + '.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['n', 'coeffs'])
            
        with open('optimal' + str(n) + '.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            func_string = '|'.join(map(str, func))
            writer.writerow([n, func_string])
    
if __name__ == "__main__":
    # reads options from config file
    config = ConfigParser()
    config.read('config.ini')
    n = config.get('classify', 'n')

    #try:
    n = int(n) # ensures that n is an integer
    main(n)
    #except:
    #    print('Error - min and max must be integers')