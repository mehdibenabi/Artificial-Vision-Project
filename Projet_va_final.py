import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from glob import glob
import open3d as o3d

# Configuration
shrink = 0.2  # Facteur de redimensionnement pour accélérer le traitement
visualize_steps = True  # Afficher les étapes intermédiaires

def load_images(image_dir):
    """Charge toutes les images d'un répertoire dans l'ordre."""
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
    if not image_paths:
        image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
    
    if not image_paths:
        raise FileNotFoundError(f"Aucune image trouvée dans {image_dir}")
    
    print(f"Chargement de {len(image_paths)} images...")
    
    images = []
    for path in image_paths:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (0, 0), fx=shrink, fy=shrink, interpolation=cv.INTER_CUBIC)
        images.append(img)
        
    return images

def detect_features(images):
    """Détecte les points SIFT et leurs descripteurs pour toutes les images."""
    sift = cv.SIFT_create()
    
    all_keypoints = []
    all_descriptors = []
    
    for i, img in enumerate(images):
        kp, des = sift.detectAndCompute(img, None)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        
        if visualize_steps:
            # Visualiser les points clés
            img_sift = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imshow(f"SIFT Keypoints - Image {i}", img_sift)
            cv.imwrite(f"sift_keypoints_{i}.png", img_sift)
    
    return all_keypoints, all_descriptors

def match_features(descriptors1, descriptors2):
    """Met en correspondance les descripteurs entre deux images."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Filtrer les bonnes correspondances selon le critère de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches

def get_matched_points(keypoints1, keypoints2, matches):
    """Extrait les points correspondants à partir des keypoints et des matches."""
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def visualize_matches(img1, kp1, img2, kp2, matches, name="matches"):
    """Visualise les correspondances entre deux images."""
    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:100], None, 
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow(f"Matches - {name}", match_img)
    cv.imwrite(f"matches_{name}.png", match_img)

def estimate_pose(K, pts1, pts2):
    """Estime la pose relative (R, t) entre deux vues."""
    # Calculer la matrice essentielle
    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    
    # Récupérer uniquement les inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    # Décomposer la matrice essentielle pour obtenir R et t
    _, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    
    return R, t, pts1, pts2, mask

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """Triangule les points 3D à partir de deux vues."""
    # Construire les matrices de projection
    P1 = np.hstack((R1, t1))
    P2 = np.hstack((R2, t2))
    
    P1 = K @ P1
    P2 = K @ P2
    
    # Triangulation
    pts1_homogeneous = cv.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_homogeneous = cv.convertPointsToHomogeneous(pts2)[:, 0, :]
    
    points_4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = cv.convertPointsFromHomogeneous(points_4d.T)[:, 0, :]
    
    return points_3d

def main():
    # Charger les images
    images = load_images("./images2")  
    
    if len(images) < 2:
        print("Au moins deux images sont nécessaires pour la reconstruction.")
        return
    
    # Détecter les caractéristiques SIFT pour toutes les images
    all_keypoints, all_descriptors = detect_features(images)
    
    # Estimer la matrice de calibration (à adapter selon votre caméra)
    # Pour une caméra non calibrée, on peut utiliser une approximation
    h, w = images[0].shape
    focal_length = 0.8 * w  # Approximation de la longueur focale

    K = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])
    
    # K = np.array([
    #  [1596,    0, 960],
    #  [   0, 1596, 540],
    #  [   0,    0,   1]
    # ])
    
    # Initialiser la reconstruction avec les deux premières images
    R_list = [np.eye(3)]  # Rotation de la première caméra (identité)
    t_list = [np.zeros((3, 1))]  # Translation de la première caméra (origine)
    
    all_points_3d = []
    all_point_colors = []
    
    # Traiter les paires d'images consécutives
    for i in range(len(images) - 1):
        print(f"Traitement des images {i} et {i+1}...")
        
        # Mettre en correspondance les caractéristiques
        matches = match_features(all_descriptors[i], all_descriptors[i+1])
        
        if visualize_steps:
            visualize_matches(images[i], all_keypoints[i], 
                             images[i+1], all_keypoints[i+1], 
                             matches, f"{i}_{i+1}")
        
        # Obtenir les points correspondants
        pts1, pts2 = get_matched_points(all_keypoints[i], all_keypoints[i+1], matches)
        
        if len(pts1) < 8:
            print(f"Pas assez de correspondances entre les images {i} et {i+1}")
            continue
        
        # Estimer la pose relative
        R_rel, t_rel, pts1_inliers, pts2_inliers, mask = estimate_pose(K, pts1, pts2)
        
        # Calculer la pose absolue de la caméra i+1
        R_abs = R_list[i] @ R_rel
        t_abs = R_list[i] @ t_rel + t_list[i]
        
        R_list.append(R_abs)
        t_list.append(t_abs)
        
        # Triangulation des points 3D
        points_3d = triangulate_points(K, R_list[i], t_list[i], R_abs, t_abs, pts1_inliers, pts2_inliers)
        
        # Filtrer les points avec une profondeur négative ou trop éloignés
        valid_points = []
        for j, pt in enumerate(points_3d):
            # Vérifier que le point est devant les deux caméras
            if pt[2] > 0:
                valid_points.append(pt)
                # Utiliser la valeur de gris comme couleur (ou vous pourriez utiliser des images couleur)
                all_point_colors.append([128, 128, 128])  # Gris
        
        all_points_3d.extend(valid_points)
        print(f"Ajout de {len(valid_points)} points 3D valides")
    
    # Visualiser le nuage de points avec Open3D
    if all_points_3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points_3d))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_point_colors) / 255.0)
        
        # Visualiser les positions des caméras
        camera_centers = []
        for i, (R, t) in enumerate(zip(R_list, t_list)):
            # Le centre de la caméra en coordonnées mondiales est -R^T * t
            C = -R.T @ t
            camera_centers.append(C.flatten())
        
        # Créer un nuage de points pour les centres des caméras
        camera_pcd = o3d.geometry.PointCloud()
        camera_pcd.points = o3d.utility.Vector3dVector(np.array(camera_centers))
        camera_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(camera_centers)))
        
        # Sauvegarder le nuage de points
        o3d.io.write_point_cloud("reconstruction.ply", pcd)
        
        # Visualiser
        o3d.visualization.draw_geometries([pcd, camera_pcd])
    else:
        print("Aucun point 3D valide n'a été reconstruit.")

if __name__ == "__main__":
    main()
    cv.waitKey(0)
    cv.destroyAllWindows()