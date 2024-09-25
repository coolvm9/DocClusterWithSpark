# DocClusterWithSpark


Components:

	1.	text_embeddings:
	•	This is an array of embeddings generated for the textual data (e.g., product reviews). Each row represents the vectorized form of a review. If you are using a pre-trained transformer model, the dimension of each embedding might be, for example, 384.
	•	Shape: (number_of_rows, embedding_dimension), e.g., (1000, 384) for 1000 reviews with 384-dimensional embeddings.
	2.	category_encoded:
	•	This is the one-hot encoded array for categorical data (e.g., product categories like “Electronics”, “Home Appliances”). One-hot encoding converts categorical values into binary vectors. For example, if you have 5 unique categories, each category would be represented by a vector with 5 values where one value is 1, and the rest are 0.
	•	Shape: (number_of_rows, number_of_categories), e.g., (1000, 5) for 1000 rows and 5 different categories.
	3.	numerical_data:
	•	This contains numerical metadata (e.g., product price, rating). These are typically standardized or normalized using techniques like StandardScaler before being used.
	•	Shape: (number_of_rows, number_of_numerical_features), e.g., (1000, 2) for 1000 rows and 2 numerical features (price and rating).



The function `np.hstack([text_embeddings, category_encoded, numerical_data])` in **NumPy** is used to **horizontally stack** arrays. This means it combines multiple arrays (or vectors) side by side, along the second axis (columns).

Here's what happens in detail:

### Components:

1. **`text_embeddings`**:
    - This is an array of embeddings generated for the textual data (e.g., product reviews). Each row represents the vectorized form of a review. If you are using a pre-trained transformer model, the dimension of each embedding might be, for example, 384.
    - Shape: `(number_of_rows, embedding_dimension)`, e.g., `(1000, 384)` for 1000 reviews with 384-dimensional embeddings.

2. **`category_encoded`**:
    - This is the **one-hot encoded** array for categorical data (e.g., product categories like "Electronics", "Home Appliances"). One-hot encoding converts categorical values into binary vectors. For example, if you have 5 unique categories, each category would be represented by a vector with 5 values where one value is 1, and the rest are 0.
    - Shape: `(number_of_rows, number_of_categories)`, e.g., `(1000, 5)` for 1000 rows and 5 different categories.

3. **`numerical_data`**:
    - This contains **numerical metadata** (e.g., product price, rating). These are typically standardized or normalized using techniques like `StandardScaler` before being used.
    - Shape: `(number_of_rows, number_of_numerical_features)`, e.g., `(1000, 2)` for 1000 rows and 2 numerical features (price and rating).

### `np.hstack` Operation:
- **`np.hstack`** horizontally stacks these arrays together, combining them side by side. It concatenates the columns of the arrays, ensuring that each row contains features from all the different sources (text embeddings, category encoding, and numerical data).

For example, let's say you have:
- `text_embeddings`: `(1000, 384)` — 384 features from text embeddings.
- `category_encoded`: `(1000, 5)` — 5 binary features from the one-hot encoding.
- `numerical_data`: `(1000, 2)` — 2 numerical features (price, rating).

The resulting array after `np.hstack([text_embeddings, category_encoded, numerical_data])` will have:
- Shape: `(1000, 391)` — 391 features for each row, where:
    - First 384 columns represent the text embeddings,
    - The next 5 columns represent the one-hot encoded categories,
    - The last 2 columns represent the numerical features (price and rating).

### Visual Representation:
| **Text Embeddings** (384-d) | **Category Encoding** (5-d) | **Numerical Data** (2-d) |
|:----------------------------|:----------------------------|:------------------------|
| [embedding vector]           | [1, 0, 0, 0, 0]             | [49.99, 5]              |
| [embedding vector]           | [0, 1, 0, 0, 0]             | [25.00, 1]              |
| ...                          | ...                         | ...                     |

### Purpose:
By combining these arrays, you are creating a **single feature matrix** that contains both:
- **Textual information** from the embeddings,
- **Categorical information** from the encoded categories,
- **Numerical information** like prices or ratings.

This combined feature matrix can now be used for various machine learning tasks, like clustering, classification, or regression.

For example, in a **clustering algorithm** (like KMeans), all these features would be treated as the input for finding similar groups or clusters.