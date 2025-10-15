import json
from pathlib import Path

import matplotlib.pyplot as plt

from .dataset import LawCorpus, TrainDataset


def analyse_law_corpus():
    law_corpus = LawCorpus(Path("../data/law_db"))

    num_images_list = []
    num_tables_list = []

    for i in law_corpus.walk_through():
        num_images_list.append(len(i.get("article_images", [])))
        num_tables_list.append(len(i.get("article_tables", [])))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    _, _, bars = axes[0].hist(
        num_images_list,
        color="skyblue",
        edgecolor="black",
        align="left",
    )
    axes[0].set_title("Number of Images per Article")
    axes[0].set_xlabel("Number of Images")
    axes[0].set_ylabel("Number of Articles")
    axes[0].bar_label(bars, fontsize=8, color="black")

    _, _, bars = axes[1].hist(
        num_tables_list,
        color="salmon",
        edgecolor="black",
        align="left",
    )
    axes[1].set_title("Number of Tables per Article")
    axes[1].set_xlabel("Number of Tables")
    axes[1].set_ylabel("Number of Articles")
    axes[1].bar_label(bars, fontsize=8, color="black")

    plt.tight_layout()
    # plt.show()
    plt.savefig("assets/law_corpus_analysis.png", dpi=300, bbox_inches="tight")


def analyze_train_dataset():
    law_corpus = LawCorpus(Path("../data/law_db"))
    train_set = TrainDataset(Path("../data/train_data"), law_corpus)

    num_articles_list = []
    for i in range(len(train_set)):
        num_articles_list.append(len(train_set[i]["relevant_articles"]))

    num_articles_list = [i for i in num_articles_list if i > 0]

    plt.figure(figsize=(8, 4))
    _, _, bars = plt.hist(num_articles_list, color="skyblue", edgecolor="black", align="left", bins=range(1, 9))
    plt.title("Number of Relevant Articles per Training Sample")
    plt.xlabel("Number of Relevant Articles")
    plt.ylabel("Number of Training Samples")
    plt.bar_label(bars, fontsize=8, color="black")

    plt.tight_layout()
    # plt.show()
    plt.savefig("assets/train_dataset_analysis.png", dpi=300, bbox_inches="tight")


def validate_train_dataset():
    law_corpus = LawCorpus(Path("../data/law_db"))
    train_set = TrainDataset(Path("../data/train_data"), law_corpus)

    faulty_data = []
    for i in range(len(train_set)):
        data_point = train_set[i]
        if data_point["__faulty__"]:
            faulty_data.append(train_set.data[i])

    if faulty_data:
        # save to a JSON file
        with open("assets/faulty_train_data.json", "w", encoding="utf-8") as f:
            json.dump(faulty_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    analyse_law_corpus()
    # analyze_train_dataset()
    # validate_train_dataset()
    pass
