import cv2 as cv
import numpy as np
import glob
import open3d as o3d

# Paramètres caméra (à ajuster selon ton téléphone)
K = np.array([
    [1596,    0, 960],
    [   0, 1596, 540],
    [   0,    0,   1]
])

# Charger les images
images = sorted(glob.glob("./images/*png"))  # ou .jpg
sift = cv.SIFT_create()
flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

Rt_global = np.eye(4)
point_cloud = []

# Image initiale
img_prev = cv.imread(images[0], cv.IMREAD_GRAYSCALE)
kp_prev, des_prev = sift.detectAndCompute(img_prev, None)

for i in range(1, len(images)):
    print(f"[INFO] Traitement image {i}/{len(images)-1}")
    img_curr = cv.imread(images[i], cv.IMREAD_GRAYSCALE)
    kp_curr, des_curr = sift.detectAndCompute(img_curr, None)

    matches = flann.knnMatch(des_prev, des_curr, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

    pts1 = np.float32([kp_prev[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp_curr[m.trainIdx].pt for m in good])

    E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.0)
    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)

    # Triangulation
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))
    pts4d = cv.triangulatePoints(P0, P1, pts1.T, pts2.T)
    pts3d = (pts4d / pts4d[3])[:3].T

    # Transformer dans le repère monde
    Rt_local = np.eye(4)
    Rt_local[:3, :3] = R
    Rt_local[:3, 3] = t.flatten()
    Rt_global = Rt_global @ np.linalg.inv(Rt_local)

    pts3d_world = (Rt_global[:3, :3] @ pts3d.T + Rt_global[:3, 3:4]).T
    point_cloud.append(pts3d_world)

    kp_prev, des_prev = kp_curr, des_curr
    img_prev = img_curr

# Assembler et visualiser
point_cloud = np.concatenate(point_cloud, axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pcd])
