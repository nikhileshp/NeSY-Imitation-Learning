import numpy as np


def euclidean_distance(eye_pos, centroid):
    """
    Calculate Euclidean distance between eye-tracking position and object centroid.
    
    Args:
        eye_pos: tuple or array (x, y) of eye-tracking coordinates
        centroid: tuple or array (x, y) of object centroid coordinates
    
    Returns:
        float: Euclidean distance
    """
    eye_pos = np.array(eye_pos)
    centroid = np.array(centroid)
    return np.linalg.norm(eye_pos - centroid)


def calculate_attention_weight(distance, frame_width, frame_height, k=0.075):
    """
    Calculate attention weight using Gaussian function.
    
    Args:
        distance: Euclidean distance between eye position and object centroid
        frame_width: Width of the frame
        frame_height: Height of the frame
        k: Constant between 0.05 and 0.1 (default: 0.075)
    
    Returns:
        float: Attention weight
    """
    s = k * min(frame_width, frame_height)
    attention = np.exp(-(distance ** 2) / (2 * s ** 2))
    return attention


def calculate_predicate_weights(eye_pos, centroids, frame_width, frame_height, k=0.075):
    """
    Calculate attention weights for all predicates (objects).
    
    Args:
        eye_pos: tuple or array (x, y) of eye-tracking coordinates
        centroids: list of tuples/arrays containing (x, y) coordinates of object centroids
        frame_width: Width of the frame
        frame_height: Height of the frame
        k: Constant between 0.05 and 0.1 (default: 0.075)
    
    Returns:
        np.ndarray: Array of attention weights for each predicate
    """
    weights = []
    for centroid in centroids:
        dist = euclidean_distance(eye_pos, centroid)
        weight = calculate_attention_weight(dist, frame_width, frame_height, k)
        weights.append(weight)
    return np.array(weights)


def calculate_example_weight(predicate_weights, alpha=5.0):
    """
    Calculate example weight as 1 + max of all predicate weights.
    
    Args:
        predicate_weights: array of predicate weights
    
    Returns:
        float: 1 + maximum of all predicate weights
    """
    if len(predicate_weights) == 0:
        return 1.0
    return 1.0 + alpha * np.max(predicate_weights)


# Example usage
if __name__ == "__main__":
    # Example data
    eye_position = (320, 240)
    object_centroids = [(300, 250), (400, 300), (150, 180)]
    frame_w, frame_h = 640, 480
    
    # Calculate predicate weights
    pred_weights = calculate_predicate_weights(
        eye_position, object_centroids, frame_w, frame_h, k=0.075
    )
    
    print("Predicate weights:", pred_weights)
    
    # Calculate example weight
    example_weight = calculate_example_weight(pred_weights)
    print("Example weight:", example_weight)
