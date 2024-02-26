from annoy import AnnoyIndex


def build_ann_index(num_trees, vectors, vector_dim=1536):
    """
    Builds an approximate nearest neighbors index using Annoy.

    Args:
    - vector_dim (int): The dimensionality of the vectors.
    - num_trees (int): The number of trees to use in the Annoy index. More trees give higher precision when querying.
    - vectors (list of lists): The vectors to add to the index. Each vector must be a list of floats.

    Returns:
    - AnnoyIndex: The Annoy index object that can be used to perform queries.
    """
    # Create an Annoy index for the given vector dimension and metric ('angular',
    # 'euclidean', 'manhattan', 'hamming', or 'dot').
    index = AnnoyIndex(vector_dim, 'angular')

    # Add items to the index
    for i, vector in enumerate(vectors):
        index.add_item(i, vector)

    # Build the index with the specified number of trees
    index.build(num_trees)

    return index


def find_nearest_neighbors(index, query_vector, n_neighbors):
    """
    Finds the nearest neighbors for a given query vector.

    Args:
    - index (AnnoyIndex): The Annoy index object.
    - query_vector (list of floats): The query vector.
    - n_neighbors (int): The number of nearest neighbors to find.

    Returns:
    - list of tuples: Each tuple contains (index of the nearest vector, the distance to the query vector).
    """
    return index.get_nns_by_vector(query_vector, n_neighbors, include_distances=True)
