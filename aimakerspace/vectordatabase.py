import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Optional

from annoy import AnnoyIndex

from aimakerspace.keywords import get_keywords
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def search_cosine_similarity(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
) -> List[Tuple[str, float]]:
    scores = [
        (key, distance_measure(query_vector, vector))
        for key, vector in self.vectors.items()
    ]
    # print("Scores in search_cosine_similarity", scores)
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def search_ann(
        self,
        query_vector: np.array,
        k: int,
) -> List[Tuple[str, float]]:
    if self.ann_index is None:
        raise ValueError("Annoy index has not been built yet.")
    neighbour_indices, distances = self.ann_index.get_nns_by_vector(query_vector, k,
                                                                    include_distances=True)
    # print("Lengths of neighbour_indices and distances", len(neighbour_indices), len(distances))
    knn = []
    for i, index in enumerate(neighbour_indices):
        knn.append((self.chunks[index], distances[i]))
    return knn


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, vector_dim=1536,
                 number_of_trees=40):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.search_engine = search_cosine_similarity

        # The following attributes are used with the ANN search
        self.ann_index: AnnoyIndex = AnnoyIndex(vector_dim, 'angular')
        self.vectors_as_list = []  # Needed to get the vectors after using ANN
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.number_of_trees = number_of_trees

    def insert(
            self,
            key: str,
            vector: np.array,
            doc_title: Optional[str] = None,
            file_name: Optional[str] = None,
            build_tree=True,
    ) -> None:
        # Older version of the storage
        self.vectors[key] = vector
        # Storage for the Annoy search engine
        self.ann_index.add_item(
            self.ann_index.get_n_items(),
            vector
        )
        self.chunks.append(key)
        self.vectors_as_list.append(vector)
        self.metadata.append({
            "keywords": get_keywords(key, number_of_words=1, top_n=5),
            "title": doc_title,
            "file_name": file_name,
        })
        if build_tree:
            self.ann_index.build(self.number_of_trees)

    def search(self,
               query_vector: np.array,
               k: int,
               ) -> List[Tuple[str, float]]:
        return self.search_engine(self, query_vector, k)

    def search_by_text(
            self,
            query_text: str,
            k: int,
            return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    # This method has been modified to build a second, parallel storage compatible with
    # the Annoy search engine and with metadata.
    async def abuild_from_list(
            self,
            list_of_text: List[str],
            list_of_titles: Optional[List[str]] = None,
            list_of_file_names: Optional[List[str]] = None,
    ) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding, title, file_name in zip(
                list_of_text, embeddings, list_of_titles, list_of_file_names
        ):
            self.insert(text, np.array(embedding), title, file_name, build_tree=False)
        # Build the trees after all the items have been added
        self.ann_index.build(self.number_of_trees)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
