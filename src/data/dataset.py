import json
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cache
from pathlib import Path

from .clean import CleanText


class MapStyleDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        """Return the item at the specified index."""
        pass


class LawCorpus:
    def __init__(self, root_dir: Path) -> None:
        """Example:
        root_dir: "../data/law_db"
        """
        self.raw_data_file = root_dir / "vlsp2025_law.json"
        self.images_folder = root_dir / "images"

        self.data = json.load(self.raw_data_file.open("r", encoding="utf-8"))

    def _prepare_article(self, doc_id: str, doc_tile: str, article: dict):
        article["images"] = CleanText.get_images(article["text"])
        article["tables"] = CleanText.get_tables(article["text"])
        return {
            "law_id": doc_id,
            "law_title": doc_tile,
            "article_id": article["id"],
            "article_text": CleanText.remove_tables_and_images(article["text"]),
            "article_title": article["title"],
            "article_images": article["images"],
            "article_tables": article["tables"],
        }

    @cache
    def get_by(self, law_id: str, article_id: str):
        """Example:
        law_id: "QCVN 41:2024/BGTVT"
        article_id: "1"
        """
        # TODO: make this more efficient and faster
        for i in self.data:
            if law_id == i["id"]:
                for j in i["articles"]:
                    if article_id == j["id"]:
                        return self._prepare_article(i["id"], i["title"], j)
        return {}

    def walk_through(self):
        for i in self.data:
            for j in i["articles"]:
                yield self._prepare_article(i["id"], i["title"], j)


class TrainDataset(MapStyleDataset):
    def __init__(self, root_dir: Path, law_corpus: LawCorpus) -> None:
        super().__init__()
        self.raw_data_file = root_dir / "vlsp_2025_train.json"
        self.images_folder = root_dir / "train_images"

        self.data = json.load(self.raw_data_file.open("r", encoding="utf-8"))
        self.law_corpus = law_corpus

    def __len__(self) -> int:
        return len(self.data)

    @cache
    def __getitem__(self, index: int):
        data_point = deepcopy(self.data[index])
        data_point["image_path"] = str(self.images_folder.joinpath(data_point["image_id"]).with_suffix(".jpg"))
        data_point["__faulty__"] = False
        for i in data_point["relevant_articles"]:
            actual_content = self.law_corpus.get_by(i["law_id"], i["article_id"])
            if not actual_content:
                data_point["__faulty__"] = True
            i.update({k: v for k, v in actual_content.items() if k != "id"})

        # Rename "id" to "data_id" to prevent collision with "id" of databases such as Elasticsearch, and Qdrant
        data_point["data_id"] = data_point.pop("id")
        return data_point


class TestDataset(MapStyleDataset):
    def __init__(self, root_dir: Path, law_corpus: LawCorpus, task_number: int) -> None:
        super().__init__()
        self.task_number = task_number
        self.raw_data_file = root_dir / f"vlsp2025_submission_task{task_number}.json"
        self.images_folder = root_dir / "private_test_images"
        self.law_corpus = law_corpus

        self.data = json.load(self.raw_data_file.open("r", encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.data)

    @cache
    def __getitem__(self, index: int):
        data_point = deepcopy(self.data[index])
        data_point["image_path"] = str(self.images_folder.joinpath(data_point["image_id"]).with_suffix(".jpg"))
        if self.task_number == 2:
            data_point["__faulty__"] = False
            for i in data_point["relevant_articles"]:
                actual_content = self.law_corpus.get_by(i["law_id"], i["article_id"])
                if not actual_content:
                    data_point["__faulty__"] = True
                i.update({k: v for k, v in actual_content.items() if k != "id"})
        return data_point

    def index_result(self, index: int, **kwargs):
        if self.task_number == 1:
            relevant_articles = kwargs.get("relevant_articles", [])
            if relevant_articles:
                self.data[index]["relevant_articles"] = relevant_articles
                return True

        if self.task_number == 2:
            answer = kwargs.get("answer", "")
            if answer:
                self.data[index]["answer"] = answer
                return True

        return False

    def save_results(self, file_path_str: str):
        try:
            output_file = Path(file_path_str)
            output_file.write_text(json.dumps(self.data, ensure_ascii=False, indent=4))
        except Exception as e:
            traceback.print_exception(e)
            return False
        return True


if __name__ == "__main__":
    law_corpus = LawCorpus(Path("../data/law_db"))
    train_set = TrainDataset(Path("../data/train_data"), law_corpus)
    identity_function = lambda x: x  # noqa: E731
    for i in range(len(train_set)):
        identity_function(train_set[i])
