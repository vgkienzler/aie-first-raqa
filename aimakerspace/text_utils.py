import os
from typing import List
from pdfminer.high_level import extract_text


class TextExtractor:
    def __init__(self):
        self.path = ""
        self.encoding = "utf-8"
        self.extract = None

    def _extract_text_from_txt(self) -> str:
        with open(self.path, "r", encoding=self.encoding) as f:
            return f.read()

    def _extract_text_from_pdf(self) -> str:
        text = extract_text(self.path)
        return text

    def extract_from(self, path: str, encoding: str = None) -> str:
        if encoding:
            self.encoding = encoding
        self.path = path
        if os.path.isfile(path) and path.endswith(".txt"):
            self.extract = self._extract_text_from_txt
        elif os.path.isfile(path) and path.endswith(".pdf"):
            self.extract = self._extract_text_from_pdf
        else:
            raise ValueError(
                "Provided path is neither a valid directory, a .pdf nor a .txt file."
            )
        return self.extract()


class FileLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path
        self.TextExtractor = TextExtractor()

    def _load_file(self, path=None):
        if not path:
            path = self.path
        extracted_text = self.TextExtractor.extract_from(path)
        self.documents.append(extracted_text)

    def _load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                self._load_file(os.path.join(root, file))

    def load_documents(self):
        if os.path.isdir(self.path):
            self._load_directory()
        else:
            self._load_file()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = FileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
