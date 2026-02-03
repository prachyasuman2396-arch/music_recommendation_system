Hybrid Music Recommendation System
Collaborative Filtering + Content-Based Learning (ALS + TF-IDF)


OVERVIEW

This project implements a Hybrid Music Recommendation System that intelligently combines user listening behavior with song content features to deliver accurate and personalized music recommendations.
Unlike traditional recommenders that rely on only one approach, this system integrates:
Collaborative Filtering using Alternating Least Squares (ALS)
Content-Based Filtering using TF-IDF (text features) + normalized audio features
This hybrid approach significantly improves recommendation quality, especially in cold-start and sparse interaction scenarios.


DATASET

Source: Million Song Dataset (Spotify + Last.fm)
Files Used:
Music Info.csv
Track metadata (artist, genre, tags)
Audio features (danceability, energy, tempo, etc.)
User Listening History.csv
user_id
track_id
playcount


SYSTEM ARCHITECTURE

User Listening History ──▶ Collaborative Filtering (ALS)
                                     │
                                     ▼
Song Metadata + Audio Features ──▶ Content-Based Filtering
                                     │
                                     ▼
                           Hybrid Recommendation Engine

                           
Part 1: Collaborative Filtering (ALS)
Why ALS?
Designed for implicit feedback (play counts, clicks, views)
Scales well on large sparse matrices
Learns latent user & item representations
Industry-standard (used by Spotify, Netflix)


Data Preprocessing


1. Log Transformation of Play Counts


user['log_counts'] = np.log1p(user['playcount'])
Why?
Raw playcounts are heavily skewed
Log scaling prevents power users from dominating training
Improves convergence stability in ALS


2. Categorical Encoding


user['user_codes'] = user["user_id"].astype("category").cat.codes
user['track_codes'] = user["track_id"].astype("category").cat.codes
Why?
ALS requires integer indices
Efficient memory representation


3. Sparse Interaction Matrix


interaction_matrix = csr_matrix(
    (user["log_counts"], (user['user_codes'], user['track_codes']))
)


Why Sparse Matrix?


User-item interactions are mostly empty
CSR format drastically reduces memory usage
Optimized for matrix factorization
🤖 ALS Model Training
model = AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=15
)


model.fit(interaction_matrix)
Hyperparameter Rationale:
Parameter	Reason
factors=50	Captures complex user taste patterns
regularization=0.1	Prevents overfitting
iterations=15	Balanced convergence & speed


Collaborative Recommendations

Uses learned latent factors
Recommends songs user hasn’t listened to
Personalized based on historical behavior


Part 2: Content-Based Recommendation

Motivation

Collaborative filtering fails when:
New songs have no interaction data
New users have little listening history
Content-based filtering solves this by analyzing song attributes.


Feature Engineering


1. Text Features

df["text_features"] = df[categorical_cols].fillna("").agg(" ".join, axis=1)
Includes:
Artist
Genre
Tags (top 3 only to reduce noise)
Why TF-IDF?
Weighs rare but meaningful tags higher
Reduces dominance of common words
Ideal for music metadata


2. Numeric Audio Features

   
StandardScaler()
Features include:
Danceability
Energy
Tempo
Loudness
Valence
Why Scaling?
Prevents features like tempo from overpowering others
Required for cosine similarity


Feature Fusion

content_matrix = hstack([
    tfidf_matrix,
    csr_matrix(df[numeric_cols].values)
])


Why Hybrid Content Matrix?
Combines semantic similarity (lyrics/tags)
With acoustic similarity (audio features)


Similarity Computation


cosine_similarity(content_matrix)
Cosine similarity measures directional closeness, ideal for:
High-dimensional sparse vectors
TF-IDF representations

Part 3: Hybrid Recommendation System


Why Hybrid?

Problem	Solution
Cold-start users	Content-based
Cold-start songs	Content-based
Personalization	Collaborative
Accuracy	Hybrid fusion

Hybrid Scoring Strategy
final_score = α * collaborative_score + (1 - α) * content_score
α = 0.6 → Collaborative weighted higher
Adjustable based on business needs


Hybrid Recommendation Function
Validates song existence
Fetches collaborative signals
Merges with content similarity
Returns ranked recommendations


Final Output
Top-10 recommended songs
Artist name
Hybrid relevance score


Example:
1. Champagne Supernova by Oasis – Hybrid Score: 0.9123
2. Don’t Look Back in Anger by Oasis – Hybrid Score: 0.9017
