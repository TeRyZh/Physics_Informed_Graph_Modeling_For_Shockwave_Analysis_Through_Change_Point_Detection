import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional


class DBSCANShockwaveClusterer:
    """
    DBSCAN-based shockwave clustering with RANSAC filtering.
    
    This implementation uses DBSCAN to identify candidate clusters, then applies
    RANSAC (Random Sample Consensus) to robustly fit shockwave propagation lines
    and filter outliers. This two-stage approach is more resistant to noise and
    outliers compared to basic linear regression.
    """
    
    def __init__(self,
                 eps_time: float = 3.0,
                 eps_space: float = 200.0,
                 min_samples: int = 5,
                 min_vehicles: int = 3,
                 min_shockwave_speed: float = -30.0,
                 max_shockwave_speed: float = -10.0,
                 ransac_iterations: int = 1000,
                 inlier_threshold: float = 150.0,
                 min_inlier_ratio: float = 0.6):
        """
        Initialize DBSCAN clusterer with RANSAC filtering.
        
        Args:
            eps_time: Time window for DBSCAN clustering (seconds)
            eps_space: Space window for DBSCAN clustering (feet)
            min_samples: Minimum points for a DBSCAN cluster
            min_vehicles: Minimum unique vehicles for valid cluster
            min_shockwave_speed: Minimum shockwave velocity (ft/s, negative for backward propagation)
            max_shockwave_speed: Maximum shockwave velocity (ft/s, negative for backward propagation)
            ransac_iterations: Number of RANSAC iterations for robust fitting
            inlier_threshold: Maximum distance for a point to be considered an inlier (feet)
            min_inlier_ratio: Minimum ratio of inliers to consider a valid model
        """
        self.eps_time = eps_time
        self.eps_space = eps_space
        self.min_samples = min_samples
        self.min_vehicles = min_vehicles
        self.min_shockwave_speed = min_shockwave_speed
        self.max_shockwave_speed = max_shockwave_speed
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
    
    def _ransac_fit_shockwave(self, 
                              times: np.ndarray, 
                              distances: np.ndarray) -> Tuple[Optional[float], List[int], Optional[float]]:
        """
        Perform RANSAC to find the best linear fit for shockwave propagation.
        
        RANSAC (Random Sample Consensus) is an iterative method to estimate parameters
        of a mathematical model from a set of observed data that contains outliers.
        
        Algorithm:
        1. Randomly select minimal subset (2 points) to fit a line
        2. Check if the slope is within valid shockwave speed range
        3. Count inliers (points close to the line)
        4. If enough inliers, refit the line using all inliers
        5. Keep the model with most inliers
        
        Args:
            times: Array of time values
            distances: Array of distance values
            
        Returns:
            Tuple containing:
            - Best fitted slope (or None if no valid model found)
            - List of indices of inlier points
            - Best fitted intercept (or None if no valid model found)
        """
        if len(times) < 2:
            return None, [], None
        
        n_points = len(times)
        best_slope = None
        best_intercept = None
        best_inliers = []
        best_inlier_count = 0
        
        # RANSAC iterations
        for iteration in range(self.ransac_iterations):
            # Randomly select 2 points to fit a line
            if n_points < 2:
                continue
            
            sample_indices = np.random.choice(n_points, 2, replace=False)
            sample_times = times[sample_indices]
            sample_distances = distances[sample_indices]
            
            # Skip if the sampled times are identical (would cause division by zero)
            if sample_times[0] == sample_times[1]:
                continue
            
            # Fit line to these two points: distance = slope * time + intercept
            slope = (sample_distances[1] - sample_distances[0]) / (sample_times[1] - sample_times[0])
            intercept = sample_distances[0] - slope * sample_times[0]
            
            # Check if slope is within reasonable range for shockwave speed
            # Negative slopes indicate backward propagation (upstream)
            if not (self.min_shockwave_speed <= slope <= self.max_shockwave_speed):
                continue
            
            # Calculate distances of all points to the fitted line
            predicted_distances = slope * times + intercept
            deviations = np.abs(distances - predicted_distances)
            
            # Identify inliers (points within threshold distance from the line)
            inlier_mask = deviations < self.inlier_threshold
            inlier_indices = np.where(inlier_mask)[0]
            inlier_count = len(inlier_indices)
            
            # Check if we have enough inliers
            if inlier_count < self.min_inlier_ratio * n_points:
                continue
            
            # Check if this model is better than the current best
            if inlier_count > best_inlier_count:
                # Refit the model using all inliers for better accuracy
                if len(inlier_indices) >= 2:
                    try:
                        inlier_times = times[inlier_indices]
                        inlier_distances = distances[inlier_indices]
                        
                        # Compute refined model parameters using all inliers
                        # This reduces the effect of the random sample selection
                        refined_coeffs = np.polyfit(inlier_times, inlier_distances, 1)
                        refined_slope = refined_coeffs[0]
                        refined_intercept = refined_coeffs[1]
                        
                        # Verify refined slope is still within reasonable range
                        if self.min_shockwave_speed <= refined_slope <= self.max_shockwave_speed:
                            best_slope = refined_slope
                            best_intercept = refined_intercept
                            best_inliers = inlier_indices.tolist()
                            best_inlier_count = inlier_count
                    except np.linalg.LinAlgError:
                        # If refinement fails, use original fit
                        best_slope = slope
                        best_intercept = intercept
                        best_inliers = inlier_indices.tolist()
                        best_inlier_count = inlier_count
                else:
                    # Not enough inliers for refinement, use original fit
                    best_slope = slope
                    best_intercept = intercept
                    best_inliers = inlier_indices.tolist()
                    best_inlier_count = inlier_count
        
        # Check if we found a valid model
        if best_inlier_count >= self.min_inlier_ratio * n_points and best_slope is not None:
            return best_slope, best_inliers, best_intercept
        else:
            return None, [], None
    
    def cluster(self, decel_points: List[Dict]) -> Tuple[List[int], Dict]:
        """
        Cluster deceleration points using DBSCAN followed by RANSAC filtering.
        
        Two-stage process:
        1. DBSCAN clustering to identify spatiotemporal groups
        2. RANSAC filtering within each cluster to:
           - Robustly estimate shockwave propagation speed
           - Remove outlier points
           - Validate cluster coherence
        
        Args:
            decel_points: List of deceleration point dictionaries with keys:
                         'time', 'distance', 'velocity', 'vehicle_id', 'trajectory_id', 'point_idx'
            
        Returns:
            Tuple of (cluster_labels, diagnostics)
            - cluster_labels: List of cluster IDs (-1 for noise/invalid)
            - diagnostics: Dictionary with clustering statistics
        """
        if not decel_points:
            return [], {}
        
        # Prepare points for clustering
        points_array = np.array([[p['time'], p['distance']] for p in decel_points])
        
        # Normalize features for DBSCAN
        # This ensures time and space dimensions contribute equally to distance calculations
        time_norm = points_array[:, 0] / self.eps_time
        space_norm = points_array[:, 1] / self.eps_space
        points_norm = np.column_stack((time_norm, space_norm))
        
        # Perform DBSCAN clustering
        # eps=1.0 because we normalized by the actual thresholds
        clusterer = DBSCAN(eps=1.0, min_samples=self.min_samples)
        dbscan_labels = clusterer.fit_predict(points_norm)
        
        # Get unique cluster IDs (excluding noise label -1)
        unique_labels = set(dbscan_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        # Initialize final labels (all start as noise)
        final_labels = np.full(len(decel_points), -1, dtype=int)
        
        # Track cluster validation statistics
        cluster_stats = {
            'total_dbscan_clusters': len(unique_labels),
            'valid_clusters': 0,
            'rejected_vehicle_count': 0,
            'rejected_ransac': 0,
            'cluster_details': []
        }
        
        # Validate and filter each DBSCAN cluster
        valid_cluster_id = 0
        
        for label in unique_labels:
            # Get points in this cluster
            cluster_mask = dbscan_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_points = [decel_points[i] for i in cluster_indices]
            
            # Criterion 1: Check vehicle diversity in cluster
            vehicles = set(p['vehicle_id'] for p in cluster_points)
            
            if len(vehicles) < self.min_vehicles:
                cluster_stats['rejected_vehicle_count'] += 1
                continue
            
            # Criterion 2: Apply RANSAC to validate shockwave velocity
            times = np.array([p['time'] for p in cluster_points])
            distances = np.array([p['distance'] for p in cluster_points])
            
            if len(times) < 2:
                cluster_stats['rejected_ransac'] += 1
                continue
            
            # Perform RANSAC fitting
            best_slope, best_inliers, best_intercept = self._ransac_fit_shockwave(times, distances)
            
            # Check if RANSAC found a valid model
            if best_slope is None or not best_inliers:
                cluster_stats['rejected_ransac'] += 1
                continue
            
            # Valid cluster found - assign labels only to inlier points
            inlier_ratio = len(best_inliers) / len(cluster_points)
            
            for inlier_idx in best_inliers:
                original_idx = cluster_indices[inlier_idx]
                final_labels[original_idx] = valid_cluster_id
            
            # Record cluster statistics
            cluster_stats['cluster_details'].append({
                'cluster_id': valid_cluster_id,
                'n_points': len(best_inliers),
                'n_vehicles': len(vehicles),
                'shockwave_speed': best_slope,
                'intercept': best_intercept,
                'inlier_ratio': inlier_ratio,
                'temporal_extent': (float(times[best_inliers].min()), 
                                   float(times[best_inliers].max())),
                'spatial_extent': (float(distances[best_inliers].min()), 
                                  float(distances[best_inliers].max()))
            })
            
            valid_cluster_id += 1
            cluster_stats['valid_clusters'] += 1
        
        # Prepare diagnostics
        diagnostics = {
            'n_clusters': cluster_stats['valid_clusters'],
            'n_noise': int(np.sum(final_labels == -1)),
            'method': 'dbscan_ransac',
            'dbscan_clusters': cluster_stats['total_dbscan_clusters'],
            'rejected_vehicle_count': cluster_stats['rejected_vehicle_count'],
            'rejected_ransac': cluster_stats['rejected_ransac'],
            'cluster_details': cluster_stats['cluster_details']
        }
        
        return final_labels.tolist(), diagnostics


# Example usage
if __name__ == "__main__":
    # Create sample deceleration points
    np.random.seed(42)
    
    # Simulate a backward-propagating shockwave (negative slope)
    # True shockwave: distance = -20 * time + 1000
    n_points = 50
    true_slope = -20  # ft/s backward propagation
    true_intercept = 1000  # feet
    
    # Generate inlier points (points on or near the shockwave line)
    times = np.linspace(0, 30, n_points)
    distances = true_slope * times + true_intercept + np.random.normal(0, 50, n_points)
    
    # Add some outlier points
    n_outliers = 10
    outlier_times = np.random.uniform(0, 30, n_outliers)
    outlier_distances = np.random.uniform(200, 1200, n_outliers)
    
    times = np.concatenate([times, outlier_times])
    distances = np.concatenate([distances, outlier_distances])
    
    # Create deceleration points list
    decel_points = []
    for i, (t, d) in enumerate(zip(times, distances)):
        decel_points.append({
            'time': t,
            'distance': d,
            'velocity': 15.0,  # Example velocity
            'vehicle_id': i % 15,  # Simulate 15 different vehicles
            'trajectory_id': i % 15,
            'point_idx': i
        })
    
    # Initialize clusterer
    clusterer = DBSCANShockwaveClusterer(
        eps_time=5.0,
        eps_space=200.0,
        min_samples=5,
        min_vehicles=3,
        min_shockwave_speed=-30.0,
        max_shockwave_speed=-10.0,
        ransac_iterations=1000,
        inlier_threshold=150.0,
        min_inlier_ratio=0.6
    )
    
    # Perform clustering
    labels, diagnostics = clusterer.cluster(decel_points)
    
    # Print results
    print("Clustering Results:")
    print(f"Total points: {len(decel_points)}")
    print(f"Valid clusters found: {diagnostics['n_clusters']}")
    print(f"Noise points: {diagnostics['n_noise']}")
    print(f"DBSCAN initial clusters: {diagnostics['dbscan_clusters']}")
    print(f"Rejected (vehicle count): {diagnostics['rejected_vehicle_count']}")
    print(f"Rejected (RANSAC): {diagnostics['rejected_ransac']}")
    
    if diagnostics['cluster_details']:
        print("\nCluster Details:")
        for cluster_info in diagnostics['cluster_details']:
            print(f"\nCluster {cluster_info['cluster_id']}:")
            print(f"  Points: {cluster_info['n_points']}")
            print(f"  Vehicles: {cluster_info['n_vehicles']}")
            print(f"  Shockwave speed: {cluster_info['shockwave_speed']:.2f} ft/s")
            print(f"  Inlier ratio: {cluster_info['inlier_ratio']:.2%}")
            print(f"  Duration: {cluster_info['temporal_extent'][1] - cluster_info['temporal_extent'][0]:.1f} s")
