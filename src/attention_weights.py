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


def calculate_attention_weight(distance, frame_width, frame_height, k=0.075, inverse=True):
    """
    Calculate attention weight using Gaussian function.
    
    Args:
        distance: Euclidean distance between eye position and object centroid
        frame_width: Width of the frame
        frame_height: Height of the frame
        k: Constant between 0.05 and 0.1 (default: 0.075)
        inverse: If True, use inverse distance formula s/(distance+s) instead of Gaussian.
                 Default False to use Gaussian which provides better discrimination.
    
    Returns:
        float: Attention weight
    """
    s = k * min(frame_width, frame_height)
    attention = np.exp(-(distance ** 2) / (2 * s ** 2))
    if inverse:
        s = 0.75 * min(frame_width, frame_height)
        attention = s/(distance+s)
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
        dist = euclidean_distance(eye_pos, centroid[1])
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


def create_object_weight_mapping(detected_objects, object_types, predicate_weights):
    """
    Create a mapping from object_id to attention weight.
    
    Args:
        detected_objects: Dictionary mapping object types to lists of GameObjects
        object_types: List of object types corresponding to predicate_weights (in same order)
        predicate_weights: Array of attention weights for each object
    
    Returns:
        dict: Mapping from object_id to weight
    """
    weight_map = {}
    
    # Flatten detected objects and create mapping
    weight_idx = 0
    for obj_type in object_types:
        obj_list = detected_objects.get(obj_type, [])
        for obj in obj_list:
            if weight_idx < len(predicate_weights):
                weight_map[obj.object_id] = predicate_weights[weight_idx]
                weight_idx += 1
    
    return weight_map


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
