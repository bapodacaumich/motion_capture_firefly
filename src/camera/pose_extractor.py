import cv2
import numpy as np
import gtsam
from gtsam import symbol_shorthand

# Shorthand for GTSAM Variable Keys
# C(i) = Camera Pose i
# L(j) = Landmark (Point) j
C = symbol_shorthand.C
L = symbol_shorthand.L

class PoseExtractor:
    """
    A class to estimate camera poses and 3D structure from a sequence of feature matches
    using Factor Graphs (GTSAM) for Bundle Adjustment.

    Key Features:
    - Supports N-camera sequences.
    - Handles 'Loop Closure' via explicit track management.
    - Uses GTSAM (Levenberg-Marquardt) for state-of-the-art manifold optimization.
    - Robust to rotation singularities (Gimbal Lock).

    -------------------------------------------------------------------------
    HOW TO USE:
    -------------------------------------------------------------------------
    1. Feature Tracking:
       You must track features across frames *before* using this class. 
       Assign a unique integer ID (track_id) to every distinct physical 3D point 
       you track.
       
       Example Chain:
       - Frame 0 & 1 match: Feature A (ID=100) is at (10,10) in Fr0 and (15,10) in Fr1.
       - Frame 1 & 2 match: Feature A (ID=100) is at (15,10) in Fr1 and (20,10) in Fr2.
       
       *Crucial*: You must pass ID=100 for both pairs so the solver knows 
       it is the same physical point.

    2. Data Format:
       Prepare a list of tuples, where each tuple represents a pair of frames:
       matches = [
           (pts0, pts1, track_ids_01),  # Pair (Cam0, Cam1)
           (pts1, pts2, track_ids_12),  # Pair (Cam1, Cam2)
           ...
       ]

    3. Run:
       extractor = PoseExtractor(K)
       cameras, points = extractor.process_sequence(matches)
    """

    def __init__(self, K=None):
        """
        Args:
            K (np.ndarray, optional): 3x3 Camera Intrinsic Matrix. 
                                      If None, a generic estimation is used.
        """
        self.K_matrix = K
        
        # State containers (Initial Guesses from OpenCV)
        self.camera_params = []   # List of [rvec(3), tvec(3)] per camera
        self.points_3d = []       # List of [x, y, z] coordinates
        
        # Observation management
        # self.observations stores: [cam_index, point_index, x_pixel, y_pixel]
        self.observations = []    
        
        # Maps unique 'track_id' -> index in self.points_3d
        self.track_map = {}       
        self._obs_set = set()     

    def process_sequence(self, matches_with_tracks):
        """
        Main pipeline to process sequential matches and run Bundle Adjustment.

        Args:
            matches_with_tracks (list): A list of tuples. Each tuple contains:
                (pts_curr, pts_next, track_ids)
                
                - pts_curr (Nx2 array): (u,v) pixel coordinates in Camera i.
                - pts_next (Nx2 array): (u,v) pixel coordinates in Camera i+1.
                - track_ids (N array): Unique integer IDs for each feature match.
        
        Returns:
            final_poses (list): List of (Rotation Matrix (3x3), Translation Vector (3x1))
            final_points (np.array): (M x 3) Array of refined 3D points.
        """
        # 1. Handle Intrinsics
        if self.K_matrix is None:
            print("Warning: No K provided. Assuming 640x480 generic.")
            f = 1.2 * 640
            self.K_matrix = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]], dtype=float)

        # 2. Initialization
        self.camera_params = [np.zeros(6)]  # Cam 0 is world origin (rvec=0, tvec=0)
        curr_R = np.eye(3)
        curr_t = np.zeros((3, 1))
        
        self.points_3d = []
        self.track_map = {}
        self.observations = []
        self._obs_set = set()

        print(f"Processing sequence of {len(matches_with_tracks) + 1} cameras...")

        # 3. Iterate through pairs (Visual Odometry Initialization)
        for cam_idx, (pts_curr, pts_next, track_ids) in enumerate(matches_with_tracks):
            
            pts_curr = np.float32(pts_curr)
            pts_next = np.float32(pts_next)
            track_ids = np.array(track_ids)

            # --- Step A: Recover Relative Pose (Cam i -> Cam i+1) ---
            E, mask = cv2.findEssentialMat(pts_curr, pts_next, self.K_matrix, 
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            mask_bool = mask.ravel() == 1
            pts_c_in = pts_curr[mask_bool]
            pts_n_in = pts_next[mask_bool]
            tids_in = track_ids[mask_bool]

            _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_c_in, pts_n_in, self.K_matrix)
            
            pose_inliers = mask_pose.ravel() > 0
            pts_c_in = pts_c_in[pose_inliers]
            pts_n_in = pts_n_in[pose_inliers]
            tids_in = tids_in[pose_inliers]

            # Accumulate Global Pose
            # t_global = t_prev + R_prev * t_rel
            curr_t = curr_t + curr_R @ t_rel
            # R_global = R_prev * R_rel
            curr_R = curr_R @ R_rel

            # Store new camera parameters (Rodrigues vector + Translation)
            rvec, _ = cv2.Rodrigues(curr_R)
            self.camera_params.append(np.hstack((rvec.ravel(), curr_t.ravel())))

            # --- Step B: Triangulate & Merge Tracks ---
            P1_rel = self.K_matrix @ np.eye(3, 4)
            P2_rel = self.K_matrix @ np.hstack((R_rel, t_rel))

            pts4d = cv2.triangulatePoints(P1_rel, P2_rel, pts_c_in.T, pts_n_in.T)
            points_local = (pts4d[:3] / pts4d[3]).T

            # Transform points to Global Frame
            prev_rvec = self.camera_params[cam_idx][:3]
            prev_tvec = self.camera_params[cam_idx][3:].reshape(3,1)
            prev_R_mat, _ = cv2.Rodrigues(prev_rvec)

            # X_world = R_cam_i * X_local + t_cam_i
            points_global = (prev_R_mat @ points_local.T).T + prev_tvec.T

            for k, tid in enumerate(tids_in):
                if tid in self.track_map:
                    point_idx = self.track_map[tid]
                else:
                    self.points_3d.append(points_global[k])
                    point_idx = len(self.points_3d) - 1
                    self.track_map[tid] = point_idx

                self._add_observation(cam_idx, point_idx, pts_c_in[k])
                self._add_observation(cam_idx + 1, point_idx, pts_n_in[k])

        # 4. Run GTSAM Optimization
        print(f"Starting GTSAM Optimization on {len(self.camera_params)} cameras and {len(self.points_3d)} points...")
        return self._run_gtsam_optimization()

    def _add_observation(self, cam_idx, point_idx, uv):
        """Helper to ensure we don't duplicate observations."""
        obs_key = (cam_idx, point_idx)
        if obs_key not in self._obs_set:
            self.observations.append([cam_idx, point_idx, uv[0], uv[1]])
            self._obs_set.add(obs_key)

    def _run_gtsam_optimization(self):
        """
        Builds the Factor Graph and optimizes using Levenberg-Marquardt.
        """
        # --- 1. Setup Graph & Noise Models ---
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()
        
        # Noise: 1.0 pixel error for measurements (Robust Huber can be added here)
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        
        # Noise: Hard constraint for Cam 0 (The World Origin)
        # 6DOF noise: 0.001 rads for rotation, 0.001 m for translation
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001]*6))
        
        # Calibration Object
        fx, fy = self.K_matrix[0,0], self.K_matrix[1,1]
        cx, cy = self.K_matrix[0,2], self.K_matrix[1,2]
        s = 0.0
        K_gtsam = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # --- 2. Add Prior (Pin Cam 0) ---
        # Cam 0 is Identity.
        pose0 = gtsam.Pose3() 
        graph.add(gtsam.PriorFactorPose3(C(0), pose0, pose_prior_noise))
        initial_estimates.insert(C(0), pose0)

        # --- 3. Add Observations (Factors) ---
        for obs in self.observations:
            cam_idx, point_idx, u, v = obs
            
            # Add Projection Factor
            # This mathematically links the Camera Pose C(i) and Point L(j)
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                np.array([u, v]), 
                measurement_noise, 
                C(cam_idx), 
                L(point_idx), 
                K_gtsam
            ))

        # --- 4. Add Initial Estimates (The "Guess") ---
        # We must populate the graph with our approximate solution from OpenCV
        
        # A. Cameras (Skip Cam 0 as we added it in the Prior step)
        for i in range(1, len(self.camera_params)):
            rvec = self.camera_params[i][:3]
            tvec = self.camera_params[i][3:]
            
            # Convert OpenCV rvec to GTSAM Rot3
            R_mat, _ = cv2.Rodrigues(rvec)
            rot = gtsam.Rot3(R_mat)
            trans = gtsam.Point3(tvec)
            
            initial_estimates.insert(C(i), gtsam.Pose3(rot, trans))
            
        # B. Points
        for j, pt in enumerate(self.points_3d):
            initial_estimates.insert(L(j), gtsam.Point3(pt))

        # --- 5. Optimize ---
        print(f"Optimizing {graph.size()} factors...")
        
        # We use Levenberg-Marquardt (Batch) for high accuracy on the full sequence
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()
        
        print(f"Final Graph Error: {graph.error(result):.4f}")

        # --- 6. Extract Results ---
        final_poses = []
        final_points = []
        
        # Extract Cameras
        n_cams = len(self.camera_params)
        for i in range(n_cams):
            pose = result.atPose3(C(i))
            R = pose.rotation().matrix()
            t = pose.translation()
            final_poses.append((R, t))
            
        # Extract Points
        n_points = len(self.points_3d)
        for j in range(n_points):
            if result.exists(L(j)):
                pt = result.atPoint3(L(j))
                final_points.append([pt[0], pt[1], pt[2]])
            else:
                # Fallback if point was pruned by optimizer (rare)
                final_points.append(self.points_3d[j])

        return final_poses, np.array(final_points)

# --------------------------------------------------------------------------
# USAGE EXAMPLE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup Mock Intrinsics
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
    
    # 2. Generate Synthetic Data
    # Pair 1: Cam 0 -> Cam 1
    pts0 = np.random.rand(50, 2) * 640
    pts1 = pts0 + 10 
    ids_01 = np.arange(50) 
    
    # Pair 2: Cam 1 -> Cam 2
    # Overlap IDs 25..49 to ensure connectivity (Loop Closure)
    pts1_next = pts1 + np.random.normal(0, 1, pts1.shape)
    pts2 = pts1_next + 10
    ids_12 = np.arange(25, 75) 
    
    matches = [
        (pts0, pts1, ids_01),       # Pair 0-1
        (pts1_next, pts2, ids_12)   # Pair 1-2
    ]
    
    # 3. Run Pipeline
    extractor = PoseExtractor(K)
    poses, structure = extractor.process_sequence(matches)
    
    print("\n--- Final Results ---")
    for i, (R, t) in enumerate(poses):
        print(f"Cam {i} Position:\n{t}")