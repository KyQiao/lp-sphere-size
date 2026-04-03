import os
import numpy as np
import cvxpy as cp
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def main():
    # --- Parameters ---
    data_file = "3dRCP.csv"
    box_size = 1.0        # Periodic boundary condition box size
    cutoff_dist = 0.5     # Distance cutoff for neighbor search
    
    # --- Load Data ---
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found. Please ensure it is in the same directory.")
        
    print(f"Loading data from {data_file}...")
    data = np.loadtxt(data_file, delimiter=',')
    position = data[:, 0:3]
    radii = data[:, 3]
    N = len(data)
    print(f"Loaded {N} particles.")

    # --- Build Neighbor Tree ---
    # KD tree is used to build neighbor list with Periodic Boundary Conditions (PBC)
    print("Building KD-Tree and calculating distance matrix...")
    tree = cKDTree(position, boxsize=[box_size, box_size, box_size])
    disMat = tree.sparse_distance_matrix(tree, cutoff_dist).toarray()
    
    # Initialize index matrix to record particle pair distances
    # NOTE: If using multiple frames, wrap this part in a loop to obtain min r_ij
    index = 1000 * np.ones((N, N))
    index = np.minimum(index, disMat)

    # --- Construct Incidence Matrix (Eq. 9) ---
    print("Constructing incidence matrix...")
    # Find pairs where index is nonzero and less than 1 (upper triangle only to avoid duplicates)
    pos = np.where((np.triu(index, 1) > 0) & (index < 1))
    num_pairs = pos[0].size
    
    # Construct 'b' vector (distances)
    b = np.empty(num_pairs, dtype=float)
    for k, i in enumerate(zip(pos[0], pos[1])):
        b[k] = index[i]

    # The incidence matrix M is sparse, so we use csr_matrix to save memory
    row = []
    col = []
    for i, p in enumerate(zip(pos[0], pos[1])):
        row.extend([i, i])
        col.extend([p[0], p[1]])
        
    M = csr_matrix((np.ones(num_pairs * 2), (row, col)), shape=(num_pairs, N))

    # --- Linear Programming (CVXPY) ---
    print("Setting up and solving Linear Programming problem...")
    r = cp.Variable(N)          # Variable r in Eq. 9
    c = np.ones(N)              # Objective function coefficients
    
    # Maximize sum of radii subject to non-negativity and spatial constraints
    prob = cp.Problem(cp.Maximize(c.T @ r), [r >= 0, M @ r <= b])
    prob.solve()
    
    # --- Plotting Results ---
    print("Generating plot...")
    plt.figure(figsize=(6, 5))
    plt.plot(radii, r.value / radii, '*', markersize=6, alpha=0.7)
    
    # Formatting
    plt.ylim(0.999, 1.001)
    plt.xlabel(r"Original Radius ($r_i$)", fontsize=12)
    plt.ylabel(r"Ratio ($r'_i / r_i$)", fontsize=12)
    plt.title("LP Result: 3D RCP Sample", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save and show
    plt.savefig("result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
