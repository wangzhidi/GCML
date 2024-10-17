import numpy as np
import os
import shutil

def get_rigid_transform(A, B):
    """
    Get rotation and translation matrix that can transform B to A.
    For most of the time, A and B should have the shape of [22,3]
    """
    assert A.shape == B.shape
    # caculate the center
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # substract
    AA = A - centroid_A
    BB = B - centroid_B

    # get rotation matrix with SVD
    H = np.dot(BB.T, AA)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # check reflections and make corrections
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # get translation matrix
    t = centroid_A.T - np.dot(R, centroid_B.T)

    return R, t

def local2global(matrix, R, t):
    """
    apply R and t to a local coordinate matrix, which should be [22,3,n_frames]
    """
    transformed_matrix = np.copy(matrix)
    p, q, r = matrix.shape
    for j in range(r):
        sub_matrix = matrix[:, :, j]
        sub_matrix_transformed = np.dot(sub_matrix, R.T) + t.T
        transformed_matrix[:, :, j] = sub_matrix_transformed
    return transformed_matrix

def global2local(matrix, R, t):
    """
    apply R and t to a global coordinate matrix, which should be [22,3,n_frames], use the same R and t as local2global
    """
    transformed_matrix = np.copy(matrix)
    p, q, r = matrix.shape
    for j in range(r):
        sub_matrix = matrix[:, :, j]
        sub_matrix_transformed = np.dot(sub_matrix - t.T, R)
        transformed_matrix[:, :, j,] = sub_matrix_transformed
    return transformed_matrix


def gather_files(source_dirs, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    file_counter = 0
    
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            files.sort()
            for file in files:
                source_file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1]
                destination_file_path = os.path.join(destination_dir, f"frame{file_counter:04d}{file_extension}")
                shutil.copy2(source_file_path, destination_file_path)
                file_counter += 1