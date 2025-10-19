import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd
import pickle as pk

# ================================
# 🔸 Helper function to normalize values
# ================================
def normalize_value(val):
    """
    Convert any input value into an integer suitable for encryption.
    Handles:
        - 'f'/'F' -> 0
        - 't'/'T' -> 1
        - '?' or empty string or NaN -> 0
        - Numeric values as they are
    """
    if pd.isna(val):  # Handle NaN
        return 0

    if isinstance(val, str):
        val = val.strip()
        if val in ["", "?", "NA", "nan", "NaN"]:
            return 0

        mapping = {'f': 0, 'F': 0, 't': 1, 'T': 1, 'M': 1, 'm': 1}
        if val in mapping:
            return mapping[val]

        try:
            return int(float(val))
        except ValueError:
            return 0  # Fallback if conversion fails

    try:
        return int(val)
    except:
        return 0

def polymul(x, y, modulus, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus))

def polyadd(x, y, modulus, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus))

def gen_binary_poly(size):
    sk = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    return np.asarray(sk[:size])

def gen_uniform_poly(size, modulus):
    a = [33069, 14354, 61150, 5079, 4813, 61939, 48212, 15209, 1916, 64176, 15340, 34874,
         23392, 5500, 44638, 8577, 60549, 13446, 2196, 12972, 30699, 60878, 61096, 63875,
         25446, 56251, 56956, 61123, 10296, 13870, 3770, 5442]
    return np.asarray(a[:size])

def gen_normal_poly(size):
    e = [0, 1, 0, 1, 0, -1, 0, 1, 0, -2, -1, 1, -1, 4, 3, -1, -2, 1, -3, -1, -2, 0, 1, 2,
         -3, -3, 0, 0, 0, -1, 0, 2]
    return np.asarray(e[:size])

# ================================
# 🔸 Key Generation
# ================================
def keygen(size, modulus, poly_mod):
    sk = gen_binary_poly(size)
    a = gen_uniform_poly(size, modulus)
    e = gen_normal_poly(size)
    b = polyadd(polymul(-a, sk, modulus, poly_mod), -e, modulus, poly_mod)
    return (b, a), sk

# ================================
# 🔸 Encryption / Decryption
# ================================
def encrypt(pk, size, q, t, poly_mod, pt):
    pt = normalize_value(pt)
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = (delta * m) % q
    e1 = gen_normal_poly(size)
    e2 = gen_normal_poly(size)
    u = gen_binary_poly(size)
    ct0 = polyadd(polyadd(polymul(pk[0], u, q, poly_mod), e1, q, poly_mod), scaled_m, q, poly_mod)
    ct1 = polyadd(polymul(pk[1], u, q, poly_mod), e2, q, poly_mod)
    return (ct0, ct1)

def decrypt(sk, size, q, t, poly_mod, ct):
    scaled_pt = polyadd(polymul(ct[1], sk, q, poly_mod), ct[0], q, poly_mod)
    decrypted_poly = np.round(scaled_pt * t / q) % t
    return int(decrypted_poly[0])

def add_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    pt = normalize_value(pt)
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = (delta * m) % q
    new_ct0 = polyadd(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])

def mul_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    pt = normalize_value(pt)
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)

# ================================
# 🔸 Encryption of Dataset
# ================================
def privacyPreservingTrain(inputFile, outputFile):
    n = 2**5
    q = 2**16
    t = 2**10
    poly_mod = np.array([1] + [0] * (n - 1) + [1])

    pk, sk = keygen(n, q, poly_mod)
    data = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n'
    dataset = pd.read_csv(inputFile)
    dataset.fillna(0, inplace=True)
    dataset = dataset.values

    for i in range(len(dataset)):
        for j in range(0, 13):
            value = dataset[i, j]
            enc = encrypt(pk, n, q, t, poly_mod, value)
            encryptData = enc[0][0]
            data += str(encryptData) + ","
        target_value = normalize_value(dataset[i, 13])
        data += str(target_value) + "\n"

    with open(outputFile, "w") as f:
        f.write(data)

# ================================
# 🔸 Testing Encryption (Optional)
# ================================
def privacyPreservingTest(inputFile, outputFile):
    n = 2**5
    q = 2**16
    t = 2**10
    poly_mod = np.array([1] + [0] * (n - 1) + [1])

    pk, sk = keygen(n, q, poly_mod)
    data = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n'
    dataset = pd.read_csv(inputFile)
    dataset.fillna(0, inplace=True)
    dataset = dataset.values

    for i in range(len(dataset)):
        for j in range(0, 13):
            value = dataset[i, j]
            enc = encrypt(pk, n, q, t, poly_mod, value)
            encryptData = enc[0][0]
            data += str(encryptData) + ","
        data = data[:-1] + "\n"

    with open(outputFile, "w") as f:
        f.write(data)
