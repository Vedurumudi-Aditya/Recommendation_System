# Recommends items using collaborative filtering with cosine similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Generate synthetic user-item interaction matrix
np.random.seed(42)
n_users = 100
n_items = 50
interactions = np.random.randint(0, 6, size=(n_users, n_items))  # 0-5 rating scale

# Convert to DataFrame
df = pd.DataFrame(interactions, columns=[f'item_{i}' for i in range(n_items)])

# Compute similarity matrix
similarity_matrix = cosine_similarity(df)

# Function to get recommendations
def get_recommendations(user_id, matrix, sim_matrix, n_recommendations=5):
    user_ratings = matrix[user_id]
    similar_users = sim_matrix[user_id]
    similar_users_indices = np.argsort(similar_users)[::-1][1:n_recommendations + 1]
    recommendations = []
    for idx in similar_users_indices:
        unrated_items = np.where(matrix[idx] > 0)[0]
        for item in unrated_items:
            if user_ratings[item] == 0:
                recommendations.append(item)
    return list(set(recommendations))[:n_recommendations]

# Example usage
user_id = 0
recommendations = get_recommendations(user_id, df.values, similarity_matrix)
print(f"Recommendations for user {user_id}: {recommendations}")