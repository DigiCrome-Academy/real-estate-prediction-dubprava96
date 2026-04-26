"""
Phase 3: Recommendation System Module

This module implements recommendation engines for real estate properties:
- Content-Based Filtering (using property features)
- Collaborative Filtering (using simulated user-property interactions)
- Hybrid Recommendation System (combining both approaches)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Section 1: Content-Based Filtering
# =============================================================================

def compute_property_similarity(X, metric='cosine'):
    if metric == 'cosine':
        sim = cosine_similarity(X)
    elif metric == 'euclidean':
        dist = euclidean_distances(X)
        sim = 1 / (1 + dist)  # convert distance → similarity
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    return sim

    """
    Compute pairwise similarity between all properties based on their features.

    Args:
        X (np.ndarray): Scaled feature matrix (n_properties x n_features).
        metric (str): Similarity metric — 'cosine' or 'euclidean'.

    Returns:
        np.ndarray: Similarity matrix of shape (n_properties, n_properties).
            Values should be between 0 and 1 (higher = more similar).

    Example:
        >>> X = np.array([[1, 2], [1, 2.1], [5, 6]])
        >>> sim = compute_property_similarity(X, metric='cosine')
        >>> sim.shape == (3, 3)
        True
        >>> np.allclose(np.diag(sim), 1.0)  # self-similarity = 1
        True
        >>> sim[0, 1] > sim[0, 2]  # first two are more similar
        True
    """
    # TODO: Implement this function
    # Hints:
    #   - For 'cosine': use cosine_similarity from sklearn
    #   - For 'euclidean': use euclidean_distances, then convert to similarity
    #     (e.g., 1 / (1 + distance))
    #   - Ensure all values are between 0 and 1
    raise NotImplementedError("Implement compute_property_similarity()")


def content_based_recommend(property_index, similarity_matrix, n_recommendations=5):
    scores = similarity_matrix[property_index]

    # sort indices by similarity (descending)
    sorted_idx = np.argsort(scores)[::-1]

    recommendations = []
    for idx in sorted_idx:
        if idx == property_index:
            continue

        recommendations.append({
            'property_index': int(idx),
            'similarity_score': float(scores[idx])
        })

        if len(recommendations) == n_recommendations:
            break

    return recommendations

    """
    Recommend properties similar to a given property using content-based filtering.

    Args:
        property_index (int): Index of the query property.
        similarity_matrix (np.ndarray): Precomputed similarity matrix.
        n_recommendations (int): Number of recommendations to return.

    Returns:
        list[dict]: List of recommendations, each with:
            - 'property_index' (int): Index of the recommended property
            - 'similarity_score' (float): Similarity to the query property

        Sorted by similarity_score descending. Must NOT include the query property.

    Example:
        >>> sim = np.array([[1.0, 0.9, 0.3],
        ...                 [0.9, 1.0, 0.4],
        ...                 [0.3, 0.4, 1.0]])
        >>> recs = content_based_recommend(0, sim, n_recommendations=2)
        >>> len(recs) == 2
        True
        >>> recs[0]['property_index'] == 1  # most similar to property 0
        True
        >>> recs[0]['similarity_score'] == 0.9
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Get similarity scores for the given property
    #   2. Sort by descending similarity
    #   3. Exclude the query property itself
    #   4. Return top n_recommendations
    raise NotImplementedError("Implement content_based_recommend()")


def knn_recommend(X, property_index, n_recommendations=5, metric='minkowski'):
     model = NearestNeighbors(n_neighbors=n_recommendations + 1, metric=metric)
     model.fit(X)

     distances, indices = model.kneighbors([X[property_index]])

     recs = []
     for dist, idx in zip(distances[0], indices[0]):
        if idx == property_index:
            continue

        recs.append({
            'property_index': int(idx),
            'distance': float(dist)
        })
     return recs[:n_recommendations]
     
     """
    Recommend properties using K-Nearest Neighbors.

    Args:
        X (np.ndarray): Scaled feature matrix.
        property_index (int): Index of the query property.
        n_recommendations (int): Number of neighbors to return.
        metric (str): Distance metric for NearestNeighbors.

    Returns:
        list[dict]: List of recommendations, each with:
            - 'property_index' (int)
            - 'distance' (float)

    Example:
        >>> X = np.random.rand(50, 5)
        >>> recs = knn_recommend(X, property_index=0, n_recommendations=3)
        >>> len(recs) == 3
        True
        >>> all('property_index' in r and 'distance' in r for r in recs)
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Fit NearestNeighbors with n_neighbors = n_recommendations + 1
    #   2. Query for the property at property_index
    #   3. Exclude the query property from results
     raise NotImplementedError("Implement knn_recommend()")


# =============================================================================
# Section 2: Collaborative Filtering
# =============================================================================

def create_user_property_matrix(n_users=100, n_properties=500, sparsity=0.95, random_state=42):

    np.random.seed(random_state)

    # random ratings 1–5
    ratings = np.random.randint(1, 6, size=(n_users, n_properties))

    # mask for sparsity
    mask = np.random.rand(n_users, n_properties) > sparsity

    matrix = ratings * mask  # zero where mask is False

    return matrix
    """
    Create a simulated user-property interaction/rating matrix.

    This simulates user preferences for properties on a 1-5 rating scale.
    Most entries should be 0 (unrated) to simulate realistic sparsity.

    Args:
        n_users (int): Number of simulated users.
        n_properties (int): Number of properties.
        sparsity (float): Fraction of entries that are 0 (between 0 and 1).
        random_state (int): Random seed.

    Returns:
        np.ndarray: Matrix of shape (n_users, n_properties) with ratings 0-5.
            0 means unrated; 1-5 are ratings.

    Example:
        >>> matrix = create_user_property_matrix(n_users=50, n_properties=100, sparsity=0.9)
        >>> matrix.shape == (50, 100)
        True
        >>> (matrix == 0).sum() / matrix.size >= 0.85  # roughly sparse
        True
        >>> matrix.max() <= 5 and matrix.min() >= 0
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Generate random ratings (1-5) for all entries
    #   2. Create a mask where ~sparsity fraction of entries are kept
    #   3. Set the rest to 0
    raise NotImplementedError("Implement create_user_property_matrix()")


def user_based_collaborative_filter(user_property_matrix, user_index, n_recommendations=5):

     sim = cosine_similarity(user_property_matrix)
     user_sim = sim[user_index]

    # top similar users (exclude self)
     similar_users = np.argsort(user_sim)[::-1][1:11]

     target_ratings = user_property_matrix[user_index]

     predictions = []

     for item_idx in range(user_property_matrix.shape[1]):
        if target_ratings[item_idx] != 0:
            continue

        numerator = 0
        denominator = 0

        for u in similar_users:
            rating = user_property_matrix[u, item_idx]
            if rating > 0:
              numerator += user_sim[u] * rating
            denominator += user_sim[u]

        if denominator > 0:
            pred = numerator / denominator
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

     return [
        {'property_index': int(idx), 'predicted_rating': float(score)}
        for idx, score in predictions[:n_recommendations]
    ]

     """
    Recommend properties for a user using user-based collaborative filtering.

    Steps:
    1. Compute cosine similarity between the target user and all other users
    2. Find the most similar users
    3. Recommend properties that similar users rated highly but the target user hasn't rated

    Args:
        user_property_matrix (np.ndarray): User-property rating matrix (n_users x n_properties).
        user_index (int): Index of the target user.
        n_recommendations (int): Number of properties to recommend.

    Returns:
        list[dict]: Recommendations, each with:
            - 'property_index' (int)
            - 'predicted_rating' (float)

        Sorted by predicted_rating descending.

    Example:
        >>> np.random.seed(42)
        >>> matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        >>> recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        >>> len(recs) <= 5
        True
        >>> all('property_index' in r and 'predicted_rating' in r for r in recs)
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Compute cosine similarity between users
    #   2. Find top-k similar users (e.g., top 10)
    #   3. For unrated properties of target user, compute weighted average rating
    #   4. Return top-n properties by predicted rating
     raise NotImplementedError("Implement user_based_collaborative_filter()")


def item_based_collaborative_filter(user_property_matrix, user_index, n_recommendations=5):
  item_sim = cosine_similarity(user_property_matrix.T)

  user_ratings = user_property_matrix[user_index]

  predictions = []

  for item_idx in range(len(user_ratings)):
        if user_ratings[item_idx] != 0:
            continue

        numerator = 0
        denominator = 0

        for j, rating in enumerate(user_ratings):
            if rating > 0:
                sim = item_sim[item_idx, j]
                numerator += sim * rating
                denominator += abs(sim)

        if denominator > 0:
            pred = numerator / denominator
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [
        {'property_index': int(idx), 'predicted_rating': float(score)}
        for idx, score in predictions[:n_recommendations]
    ]

  """
    Recommend properties using item-based collaborative filtering.

    Steps:
    1. Compute item-item (property-property) similarity from the rating matrix
    2. For each unrated property, predict rating based on similar rated properties
    3. Return top-n predictions

    Args:
        user_property_matrix (np.ndarray): User-property rating matrix.
        user_index (int): Index of the target user.
        n_recommendations (int): Number of properties to recommend.

    Returns:
        list[dict]: Recommendations, each with:
            - 'property_index' (int)
            - 'predicted_rating' (float)

        Sorted by predicted_rating descending.

    Example:
        >>> matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        >>> recs = item_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        >>> len(recs) <= 5
        True
    """
    # TODO: Implement this function
  raise NotImplementedError("Implement item_based_collaborative_filter()")


# =============================================================================
# Section 3: Hybrid Recommendation System
# =============================================================================

def hybrid_recommend(
    property_features,
    user_property_matrix,
    user_index,
    property_index,
    content_weight=0.5,
    collaborative_weight=0.5,
    n_recommendations=5
):
    sim_matrix = compute_property_similarity(property_features)
    content_scores = sim_matrix[property_index]

    # collaborative scores (user-based)
    collab_recs = user_based_collaborative_filter(
        user_property_matrix, user_index, n_recommendations=100
    )
    collab_dict = {r['property_index']: r['predicted_rating'] for r in collab_recs}

    # normalize content
    content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
     # normalize collaborative
    if collab_dict:
        vals = np.array(list(collab_dict.values()))
        min_v, max_v = vals.min(), vals.max()
        collab_norm = {
            k: (v - min_v) / (max_v - min_v + 1e-8)
            for k, v in collab_dict.items()
        }
    else:
        collab_norm = {}

        results = []

    for i in range(len(content_scores)):
        if i == property_index:
            continue

        c_score = content_norm[i]
        col_score = collab_norm.get(i, 0)

        hybrid_score = content_weight * c_score + collaborative_weight * col_score

        results.append({
            'property_index': int(i),
            'content_score': float(c_score),
            'collaborative_score': float(col_score),
            'hybrid_score': float(hybrid_score)
        })
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return results[:n_recommendations]

    """
    Hybrid recommendation combining content-based and collaborative filtering.

    Args:
        property_features (np.ndarray): Scaled property feature matrix.
        user_property_matrix (np.ndarray): User-property rating matrix.
        user_index (int): Target user index (for collaborative).
        property_index (int): Reference property index (for content-based).
        content_weight (float): Weight for content-based scores (0 to 1).
        collaborative_weight (float): Weight for collaborative scores (0 to 1).
        n_recommendations (int): Number of final recommendations.

    Returns:
        list[dict]: Recommendations, each with:
            - 'property_index' (int)
            - 'content_score' (float)
            - 'collaborative_score' (float)
            - 'hybrid_score' (float)

        Sorted by hybrid_score descending.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        >>> recs = hybrid_recommend(X, matrix, user_index=0, property_index=10)
        >>> len(recs) <= 5
        True
        >>> all('hybrid_score' in r for r in recs)
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Get content-based similarity scores for the reference property
    #   2. Get collaborative filtering predicted ratings for the user
    #   3. Normalize both score sets to [0, 1]
    #   4. Combine: hybrid = content_weight * content + collaborative_weight * collab
    #   5. Return top-n by hybrid_score
    raise NotImplementedError("Implement hybrid_recommend()")


def evaluate_recommendations(recommendations, ground_truth_ratings, threshold=3.5):
  
 recommended_items = [r['property_index'] for r in recommendations]

 relevant_items = {
        k for k, v in ground_truth_ratings.items() if v >= threshold
    }

 recommended_relevant = [
        item for item in recommended_items if item in relevant_items
    ]

 precision = len(recommended_relevant) / len(recommended_items) if recommended_items else 0
 recall = len(recommended_relevant) / len(relevant_items) if relevant_items else 0

 return {
        'precision': precision,
        'recall': recall,
        'n_relevant_recommended': len(recommended_relevant),
        'n_recommended': len(recommended_items),
        'n_relevant_total': len(relevant_items)
    }


 """
    Evaluate recommendation quality using precision and recall.

    Args:
        recommendations (list[dict]): List of recommendation dicts with 'property_index'.
        ground_truth_ratings (dict): {property_index: actual_rating} for the user.
        threshold (float): Minimum rating to consider a property as "relevant".

    Returns:
        dict: {
            'precision': float (fraction of recs that are relevant),
            'recall': float (fraction of relevant items that are in recs),
            'n_relevant_recommended': int,
            'n_recommended': int,
            'n_relevant_total': int
        }

    Example:
        >>> recs = [{'property_index': 0}, {'property_index': 1}, {'property_index': 2}]
        >>> truth = {0: 4.0, 1: 2.0, 2: 5.0, 3: 4.5}
        >>> metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        >>> metrics['precision']  # 2 out of 3 recommended are relevant
        0.6666666666666666
        >>> metrics['recall']  # 2 out of 3 relevant are recommended
        0.6666666666666666
    """
    # TODO: Implement this function
 raise NotImplementedError("Implement evaluate_recommendations()")
