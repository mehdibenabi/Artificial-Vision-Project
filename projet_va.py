import cv2
import numpy as np
import glob
import open3d as o3d

# === PARAMÈTRES CAMERA (À ajuster) ===
focal_length = 718.8560
cx, cy = 607.1928, 185.2157
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# === CHARGEMENT DE LA SÉQUENCE ===
images = sorted(glob.glob("images/*.png"))  # ou .jpg selon ton dataset
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# === POSE INITIALE ===
Rt_global = np.eye(4)
point_cloud = []

prev_img = cv2.imread(images[0], 0)
prev_kp, prev_des = sift.detectAndCompute(prev_img, None)

for i in range(1, len(images)):
    print(f"[INFO] Traitement de la paire {i-1}-{i}")
    curr_img = cv2.imread(images[i], 0)
    curr_kp, curr_des = sift.detectAndCompute(curr_img, None)
    
    matches = bf.knnMatch(prev_des, curr_des, k=2)

    # Ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

    # Matrice essentielle
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Matrices de projection
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    # Triangulation
    pts4d_hom = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
    pts3d = (pts4d_hom / pts4d_hom[3])[:3].T

    # Transformation dans le repère monde
    Rt_local = np.eye(4)
    Rt_local[:3, :3] = R
    Rt_local[:3, 3] = t.flatten()
    Rt_global = Rt_global @ np.linalg.inv(Rt_local)

    pts3d_world = (Rt_global[:3, :3] @ pts3d.T + Rt_global[:3, 3:4]).T
    point_cloud.append(pts3d_world)

    prev_kp, prev_des = curr_kp, curr_des
    prev_img = curr_img

# === ASSEMBLER LE NUAGE FINAL ===
point_cloud = np.concatenate(point_cloud, axis=0)

# === VISUALISATION OPEN3D ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pcd])
