import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

class PoseExtractor:
    """
    A class to estimate camera poses and 3D structure from a sequence of feature matches
    using Sparse Bundle Adjustment (SBA).

    Key Features:
    - Supports N-camera sequences.
    - Handles 'Loop Closure' via explicit track management.
    - Uses Sparse Jacobians for memory-efficient optimization.

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
        self.K = K
        
        # State containers
        self.camera_params = []   # List of [rvec(3), tvec(3)] per camera
        self.points_3d = []       # List of [x, y, z] coordinates
        
        # Observation management
        # self.observations stores: [cam_index, point_index, x_pixel, y_pixel]
        self.observations = []    
        
        # Maps unique 'track_id' -> index in self.points_3d
        # This prevents creating duplicate 3D points for the same feature.
        self.track_map = {}       
        
        # Set to ensure we don't add the exact same observation twice
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
                                       If track_ids[k] == 55, then pts_curr[k] 
                                       and pts_next[k] are observations of Point 55.

        Returns:
            final_poses (list): List of (Rotation Matrix (3x3), Translation Vector (3x1))
            final_points (np.array): (M x 3) Array of refined 3D points.
        """
        # 1. Handle Intrinsics
        if self.K is None:
            print("Warning: No K provided. Assuming 640x480 generic.")
            f = 1.2 * 640
            self.K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]], dtype=float)

        # 2. Initialization
        self.camera_params = [np.zeros(6)]  # Cam 0 is world origin
        curr_R = np.eye(3)
        curr_t = np.zeros((3, 1))
        
        self.points_3d = []
        self.track_map = {}
        self.observations = []
        self._obs_set = set()

        print(f"Processing sequence of {len(matches_with_tracks) + 1} cameras...")

        # 3. Iterate through pairs (Visual Odometry)
        for cam_idx, (pts_curr, pts_next, track_ids) in enumerate(matches_with_tracks):
            
            pts_curr = np.float32(pts_curr)
            pts_next = np.float32(pts_next)
            track_ids = np.array(track_ids)

            # --- Step A: Recover Relative Pose (Cam i -> Cam i+1) ---
            E, mask = cv2.findEssentialMat(pts_curr, pts_next, self.K, 
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # Filter outliers based on Essential Matrix
            mask_bool = mask.ravel() == 1
            pts_c_in = pts_curr[mask_bool]
            pts_n_in = pts_next[mask_bool]
            tids_in = track_ids[mask_bool]

            # Recover R, t from E
            _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_c_in, pts_n_in, self.K)
            
            # Filter outliers based on Cheirality (points must be in front of camera)
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
            
            # Projection matrices for triangulation
            # P1: Relative Identity (Cam i)
            P1_rel = self.K @ np.eye(3, 4)
            # P2: Relative Pose (Cam i+1)
            P2_rel = self.K @ np.hstack((R_rel, t_rel))

            pts4d = cv2.triangulatePoints(P1_rel, P2_rel, pts_c_in.T, pts_n_in.T)
            points_local = (pts4d[:3] / pts4d[3]).T  # Shape: (N, 3) relative to Cam i

            # Transform points to Global Frame
            # We use the EXISTING global pose of Cam i
            prev_rvec = self.camera_params[cam_idx][:3]
            prev_tvec = self.camera_params[cam_idx][3:].reshape(3,1)
            prev_R_mat, _ = cv2.Rodrigues(prev_rvec)

            # X_world = R_cam_i * X_local + t_cam_i
            points_global = (prev_R_mat @ points_local.T).T + prev_tvec.T

            # Process every point in this pair
            for k, tid in enumerate(tids_in):
                
                # 1. Check if track exists (Loop Closure logic)
                if tid in self.track_map:
                    # Point already exists in our 3D world.
                    point_idx = self.track_map[tid]
                    # Optional: We could run a running average here to refine the initial guess
                else:
                    # New point. Register it.
                    self.points_3d.append(points_global[k])
                    point_idx = len(self.points_3d) - 1
                    self.track_map[tid] = point_idx

                # 2. Add Observations (Edges in the graph)
                # Cam i sees point_idx
                self._add_observation(cam_idx, point_idx, pts_c_in[k])
                
                # Cam i+1 sees point_idx
                self._add_observation(cam_idx + 1, point_idx, pts_n_in[k])

        # 4. Run Bundle Adjustment
        print(f"Starting Bundle Adjustment on {len(self.camera_params)} cameras and {len(self.points_3d)} points...")
        return self._run_sparse_bundle_adjustment()

    def _add_observation(self, cam_idx, point_idx, uv):
        """Helper to ensure we don't duplicate observations."""
        obs_key = (cam_idx, point_idx)
        if obs_key not in self._obs_set:
            self.observations.append([cam_idx, point_idx, uv[0], uv[1]])
            self._obs_set.add(obs_key)

    def _run_sparse_bundle_adjustment(self):
        """
        Constructs the sparse Jacobian and runs least_squares optimization.
        """
        n_cams = len(self.camera_params)
        n_points = len(self.points_3d)
        n_obs = len(self.observations)

        observations = np.array(self.observations)
        cam_indices = observations[:, 0].astype(int)
        point_indices = observations[:, 1].astype(int)
        points_2d = observations[:, 2:4]

        # Parameter Vector: [Camera Params (n*6) ... Point Params (m*3)]
        x0 = np.hstack((np.array(self.camera_params).ravel(), 
                        np.array(self.points_3d).ravel()))

        # --- Sparsity Matrix (Jacobian Structure) ---
        # Rows = 2 * n_obs (x and y error for each)
        # Cols = n_params
        A = lil_matrix((n_obs * 2, len(x0)), dtype=int)
        
        i = np.arange(n_obs)
        # Fill Camera block (6 params)
        for s in range(6):
            A[2 * i, cam_indices * 6 + s] = 1
            A[2 * i + 1, cam_indices * 6 + s] = 1
        
        # Fill Point block (3 params)
        for s in range(3):
            # Point params start after all camera params
            A[2 * i, n_cams * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cams * 6 + point_indices * 3 + s] = 1

        # --- Loss Function ---
        def reprojection_loss(params):
            # Unpack
            cams = params[:n_cams * 6].reshape((n_cams, 6))
            points = params[n_cams * 6:].reshape((n_points, 3))
            
            # Select params specific to observations
            obs_rvecs = cams[cam_indices, :3]
            obs_tvecs = cams[cam_indices, 3:]
            obs_points = points[point_indices]
            
            residuals = []
            
            # Compute Residuals
            # Note: A loop is used here for clarity and safety with Rodrigues.
            # For massive datasets (>50k obs), consider a custom JAX/Numba implementation.
            for k in range(n_obs):
                # 1. Rotate (World -> Cam)
                R, _ = cv2.Rodrigues(obs_rvecs[k])
                
                # 2. Translate
                # P_cam = R * P_world + t
                p_c = R @ obs_points[k] + obs_tvecs[k]
                
                # 3. Project
                # Prevent division by zero for points behind camera
                z = p_c[2] if abs(p_c[2]) > 1e-7 else 1e-7
                
                u = self.K[0,0] * (p_c[0] / z) + self.K[0,2]
                v = self.K[1,1] * (p_c[1] / z) + self.K[1,2]
                
                residuals.append(u - points_2d[k, 0])
                residuals.append(v - points_2d[k, 1])
                
            return np.array(residuals)

        # --- Optimization ---
        print("Optimizing...")
        res = least_squares(reprojection_loss, x0, jac_sparsity=A, 
                            verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss='huber')

        # --- Extract Results ---
        opt_cams_flat = res.x[:n_cams * 6].reshape((n_cams, 6))
        opt_points = res.x[n_cams * 6:].reshape((n_points, 3))
        
        final_poses = []
        for c in opt_cams_flat:
            R, _ = cv2.Rodrigues(c[:3])
            final_poses.append((R, c[3:]))
            
        return final_poses, opt_points

# --------------------------------------------------------------------------
# USAGE EXAMPLE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup Mock Intrinsics
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
    
    # 2. Generate Synthetic Data
    # Let's imagine a square room with 4 corners. 
    # We will track points on a static object in the center.
    
    # Pair 1: Cam 0 -> Cam 1
    # We track 50 points. IDs 0 to 49.
    pts0 = np.random.rand(50, 2) * 640
    pts1 = pts0 + 10 # Simulated movement
    ids_01 = np.arange(50) # IDs 0..49
    
    # Pair 2: Cam 1 -> Cam 2
    # We track 50 points. 
    # HALF are the SAME points as before (IDs 25..49), HALF are new (IDs 50..74)
    # This overlap (IDs 25-49) provides the scale constraint / loop closure.
    pts1_next = pts1 + np.random.normal(0, 1, pts1.shape) # Slightly different view in Cam 1
    pts2 = pts1_next + 10
    ids_12 = np.arange(25, 75) # IDs 25..74
    
    # Note: You must ensure pts1 (from pair 1) and pts1_next (from pair 2) 
    # refer to valid pixel coords for those IDs.
    
    # Pack the data
    # (pts_cam_i, pts_cam_j, track_IDs)
    matches = [
        (pts0, pts1, ids_01),       # Pair 0-1
        (pts1_next, pts2, ids_12)   # Pair 1-2
    ]
    
    # 3. Run Pipeline
    extractor = PoseExtractor(K)
    poses, structure = extractor.process_sequence(matches)
    
    print("\n--- Final Results ---")
    for i, (R, t) in enumerate(poses):
        print(f"Cam {i} Position:\n{t.ravel()}")