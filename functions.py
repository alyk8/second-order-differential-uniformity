import numpy as np
from numba import njit

def get_primitive_int(n): # gets the integer rep of the primitive polynomial for GF(2^n) up to n=32
    polys = {
        2: 0x7, 3: 0xB, 4: 0x13, 5: 0x25, 6: 0x43, 7: 0x83, 
        8: 0x11D, 9: 0x211, 10: 0x409, 11: 0x805, 12: 0x1053, 
        13: 0x201B, 14: 0x4443, 15: 0x8003, 16: 0x103DD, 
        17: 0x20009, 18: 0x4003F, 19: 0x80027, 20: 0x100009, 
        21: 0x200005, 22: 0x400003, 23: 0x800021, 24: 0x1000087, 
        25: 0x2000009, 26: 0x4000047, 27: 0x8000027, 28: 0x10000009, 
        29: 0x20000005, 30: 0x40800007, 31: 0x80000009, 32: 0x100400007
    }
    return polys[n]

@njit(cache=True)
def generate_gf_tables(N, MOD, poly): # generates the exponential and logarithm lookup tables for fast galois field arithmetic
    exp_table = np.zeros(N, dtype=np.int64)
    log_table = np.zeros(N, dtype=np.int64)
    val = 1
    for j in range(MOD):
        exp_table[j] = val
        log_table[val] = j
        val <<= 1 # doubles val
        if val & N: # checks if val has overflowed from the field
            val ^= poly # modulo reduction
    
    return exp_table, log_table

@njit
def get_tt(N, d1, d2, i_alpha, exp_table, log_table, MOD): # computes the truth table for the function f(x) = x^d1 + a^i x^d2
    tt = np.zeros(N, dtype=np.uint64)
    for x in range(1, N):
        log_x = log_table[x]
        res = exp_table[(log_x*d1) % MOD] # x^d1
        if d2 != -1: # if its a binomial function (i.e. x^d2 != 0)
            res ^= exp_table[(log_x*d2 + i_alpha) % MOD] # a^i_alpha x^d2
        tt[x] = res
    
    return tt

@njit
def get_anf(n, tt): # computes the ANF from a truth table using the butterfly algorithmn
    anf = tt.copy()
    for i in range(n):
        step = 1 << i # 2**i
        for j in range(0, 2**n, step*2):
            for k in range(j, j+step):
                anf[k+step] ^= anf[k]
    
    return anf

@njit
def get_kernel_basis(n, A): # finds the lexicographically smallest basis vectors for the kernel of A
    pivot_row = 0
    pivots = np.zeros(n, dtype=np.int64)
    pivots.fill(-1) # -1 = col without pivot

    # reduces matrix A to row echelon form
    for i in range(n):
        sel = -1
        for r in range(pivot_row, n):
            if (A[r] >> i) & 1: # finds a row with a 1 in the i-th col for the pivot
                sel = r
                break
        
        if sel != -1:
            A[pivot_row], A[sel] = A[sel], A[pivot_row] # swaps pivot row with the current row
            for r in range(n): # eliminates the 1s in col i for all other rows
                if r != pivot_row and ((A[r] >> i) & 1):
                    A[r] ^= A[pivot_row] # XORs the pivot row into this row
            pivots[i] = pivot_row
            pivot_row += 1

    # extracts kernel basis vectors
    basis = np.zeros(4, dtype=np.int64)
    k_idx = 1
    for i in range(n):
        if pivots[i] == -1: # if the col has no pivot
            val = 1 << i
            for j in range(n):
                if pivots[j] != -1 and ((A[pivots[j]] >> i) & 1):
                    val |= (1 << j)
            basis[k_idx] = val
            k_idx += 1
            if k_idx == 3: # stops after finding 2 basis vectors
                break

    if k_idx == 3: # the 2D subspace contains {0, basis[1], basis[2], basis[1] XOR basis[2]}
        basis[3] = basis[1] ^ basis[2]
        basis.sort() # sorts them into lexicographical order
        return basis
    else: # returns empty if the kernel dimension is less than 2
        return np.zeros(4, dtype=np.int64)

def get_exponents(n, MOD): # returns the known optimal exponent configurations [d1, d2, k] for a given dimension n<=15
    exps = [[7, -1, 0]]
    if n == 5:
        exps.append([11, -1, 0])
    elif n == 7:
        exps.append([19, -1, 0])
        exps.append([21, -1, 0])
    elif n == 8:
        exps.append([37, -1, 0])
        exps.append([19, 13*(2**3) % MOD, 1])
    elif n == 9:
        exps.append([21, -1, 0])
        exps.append([35, -1, 0])
    elif n == 10:
        exps.append([73, -1, 0])
        exps.append([7, 19*(2**6) % MOD, 31])
        exps.append([7, 25*(2**2) % MOD, 31])
        exps.append([73, 13*(2**8) % MOD, 31])
        exps.append([73, 11*(2**5) % MOD, 31])
    elif n == 11:
        exps.append([21, -1, 0])
        exps.append([67, -1, 0])
        exps.append([73, -1, 0])
        exps.append([137, -1, 0])
    elif n == 12:
        exps.append([133, -1, 0])
    elif n == 13:
        exps.append([21, -1, 0])
        exps.append([73, -1, 0])
        exps.append([131, -1, 0])
        exps.append([265, -1, 0])
        exps.append([273, -1, 0])
    elif n == 14:
        exps.append([73, -1, 0])
        exps.append([529, -1, 0])
        exps.append([7, 67*(2**8) % MOD, 127])
        exps.append([7, 97*(2**2) % MOD, 127])
        exps.append([73, 25*(2**10) % MOD, 127])
        exps.append([73, 19*(2**6) % MOD, 127])
        exps.append([529, 37*(2**2) % MOD, 127])
        exps.append([529, 41*(2**11) % MOD, 127])
    elif n == 15:
        exps.append([21, -1, 0])
        exps.append([259, -1, 0])
        exps.append([273, -1, 0])

    return np.array(exps, dtype=np.int64)

@njit
def get_AB(N, reduced=True): # gets all (a, b) pairs to be checked (reduced mode removes affine equivalence classes)
    size = int((N-1)*(N-2)/6) # the no. of pairs
    diffs = np.zeros(shape=(size, 3), dtype=np.uint32) # stores (a, a+b) pairs in vector form
    count = 0

    for a in range(1, N): # excludes the zero vector
        seen = np.zeros(N, dtype=np.uint32) # records all of the b's for this specific a
        for ab in range(a+1, N): # a < a^b
            if (not seen[ab]) or (not reduced): # if we haven't already set b to this a^b
                b = a^ab # a^(a^b) = (a^a)^b = 0^b = b
                seen[b] = 1
                if ab < b: # if we haven't already set a to this a^b
                    diffs[count, 0] = a
                    diffs[count, 1] = ab
                    diffs[count, 2] = b
                    count += 1
    
    return count, diffs

@njit
def get_rank(n, num_rows, M): # computes the rank of M over GF(2) using Gaussian elimination
    A = M.copy() # creates a copy of the matrix
    rank = 0
    for i in range(n): # for each column
        sel = -1
        for r in range(rank, num_rows): # searches for a pivot, i.e. a row with a '1' in the current column i
            if (A[r] >> i) & 1: # extracts the i-th digit of row r
                sel = r
                break # stops searching if we've found a valid pivot
        if sel != -1:
            A[rank], A[sel] = A[sel], A[rank] # swaps pivot row with current row
            for r in range(num_rows): # eliminates the 1s in col i for all other rows
                if r != rank and ((A[r] >> i) & 1):
                    A[r] ^= A[rank] # XORs the pivot row into this row
            rank += 1
    
    return rank

@njit
def get_inverse(n, M): # computes the inverse of M over GF(2) using Gauss-Jordan elimination
    A = M.copy()
    I = np.zeros(n, dtype=np.int64) # identity matrix
    for i in range(n):
        I[i] = 1 << i
    
    rank = 0
    for i in range(n):
        sel = -1
        for r in range(rank, n):
            if (A[r] >> i) & 1:
                sel = r
                break
        if sel != -1:
            A[rank], A[sel] = A[sel], A[rank]
            I[rank], I[sel] = I[sel], I[rank] # also swaps rows in matrix I
            for r in range(n):
                if r != rank and ((A[r] >> i) & 1):
                    A[r] ^= A[rank]
                    I[r] ^= I[rank]
            rank += 1
    if rank < n: # if matrix was not invertible
        return np.zeros(n, dtype=np.int64)

    return I

@njit
def get_canonical_form(n, m, coeffs): # transforms the cubic ANF coefficients into a canonical basis
    # finds a linearly independent basis from the coefficients
    basis = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(m):
        v = coeffs[i]
        temp_basis = np.zeros(count+1, dtype=np.int64)
        for j in range(count):
            temp_basis[j] = basis[j]
        temp_basis[count] = v
        if get_rank(n, count+1, temp_basis) == count+1:
            basis[count] = v
            count += 1
            if count == n:
                break

    # builds the transformation matrix B_T
    B_T = np.zeros(n, dtype=np.int64)
    for i in range(n):
        val = 0
        for j in range(n):
            if (basis[j] >> i) & 1: # extracts the i-th bit of the j-th basis vector
                val |= (1 << j)
        B_T[i] = val

    # inverts B_T to get the inverse transformation matrix
    L1 = get_inverse(n, B_T)
    gPrime = np.zeros(m, dtype=np.int64)

    # applies the inverse transformation to all coefficients
    for j in range(m):
        v = coeffs[j]
        new_v = 0 # L1*v
        for i in range(n):
            pop = 0
            temp = L1[i] & v # bitwise AND acts as element-wise multiplication in GF(2)
            while temp > 0:
                if temp & 1:
                    pop ^= 1
                temp >>= 1
            if pop:
                new_v |= (1 << i)
        gPrime[j] = new_v
        
    return gPrime

@njit
def get_cubic_indices(n, m): # gets all cubic indicies
    cubic_indices = np.zeros(m, dtype=np.uint64)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cubic_indices[count] = 2**i + 2**j + 2**k
                count += 1

    return cubic_indices

@njit
def build_M(n, a, b, M, func): # builds the matrix M for the 2D subspace
    c = 0 # coefficient no.
    
    for i in range(n):
        M[i] = 0

    # builds the matrix M for the 2D subspace
    for i in range(n): 
        ai = (a >> i) & 1
        bi = (b >> i) & 1
        for j in range(i+1, n):
            aj = (a >> j) & 1
            bj = (b >> j) & 1
            for k in range(j+1, n):
                ak = (a >> k) & 1
                bk = (b >> k) & 1

                # calculates the derivative terms
                term_a = (aj & bk) ^ (ak & bj)
                term_b = (ai & bk) ^ (ak & bi)
                term_c = (ai & bj) ^ (aj & bi)

                # if a term is 1, XOR the corresponding cubic coefficient into M
                if term_a:
                    M[i] ^= func[c]
                if term_b:
                    M[j] ^= func[c]
                if term_c:
                    M[k] ^= func[c]
                c += 1