import argparse
import glob
import os
import re
from typing import Union
from urllib import request

import fasttext.util
import MeCab
import neologdn
import numpy as np
from gensim.models.fasttext import FastText, load_facebook_model
from tqdm import tqdm


def get_data(return_path=False) -> Union[list[str], tuple[list[str], list[str]]]:
    doc_paths = []
    for dir in os.scandir("text"):
        if dir.is_dir():
            paths = sorted(
                glob.glob(os.path.join("text", dir.name, f"{dir.name}*.txt"))
            )
            doc_paths += paths

    texts = []
    for path in doc_paths:
        with open(path, "r") as f:
            texts.append("".join(f.readlines()[2:]))

    if return_path:
        return texts, doc_paths
    else:
        return texts


def get_stopword() -> list[str]:
    content = request.urlopen(
        "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    )
    stopwords = [
        line.decode("utf-8").strip() for line in content if len(line.strip()) > 0
    ]
    return stopwords


class Preprocessor:
    def __init__(self, stopwords=None, include_pos=None) -> None:
        mecab = MeCab.Tagger(
            "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
        )
        self.parser = mecab.parse
        self.stopwords = [] if stopwords is None else stopwords
        self.include_pos = ["名詞", "動詞", "形容詞"] if include_pos is None else include_pos

    def tokenize(self, text: str) -> list[str]:
        morpemes = [line.split("\t") for line in self.parser(text).split("\n")]
        tokenized_text = []
        for morpeme in morpemes:
            if len(morpeme) >= 4:
                pos = morpeme[3].split("-")[0]
                base = morpeme[2]
                if pos in self.include_pos and base not in self.stopwords:
                    tokenized_text.append(base)
        return tokenized_text

    def normalize(self, text: str) -> str:
        text = re.sub(r"\"?([-a-zA-Z0-9.`?{}]+\.jp)\"?", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[\(\),.:=\?？！\+><\[\]\|\"\';\n【】『』\-\!、。“■（）]", " ", text)
        text = neologdn.normalize(text)
        return text

    def create_dataet(self, texts: list[str]) -> list[list[str]]:
        dataset = []
        for text in tqdm(texts):
            text = self.normalize(text)
            tokenized_text = self.tokenize(text)
            dataset.append(tokenized_text)
        return dataset


class Embedder:
    def __init__(self, sentences: list[list[str]], filename="mymodel.model") -> None:
        self.model = self.create_model(sentences, filename)

    def create_model(self, sentences: list[list[str]], filename: str):
        if os.path.exists(filename):
            print("load model")
            model = FastText.load(filename)
        else:
            fasttext.util.download_model("ja", if_exists="ignore")
            model = load_facebook_model("cc.ja.300.bin.gz")
            model.build_vocab(sentences, update=True)
            model.train(sentences, total_examples=len(sentences), epochs=10)
            if filename is not None:
                model.save(filename)
        return model

    def create_embed(self, sentences: list[list[str]]) -> np.ndarray:
        embeddigs = []
        for sentence in sentences:
            embeddigs.append(np.mean([self.model.wv[w] for w in sentence], axis=0))
        return np.array(embeddigs)

    def get_similar_sentence(
        self, target_word: list[str], embed_dic: np.ndarray, topk=5
    ) -> tuple[list[int], list[float]]:
        target_vec = self.create_embed([target_word]).reshape(-1)
        similarities = np.array(
            [self.cos_sim(target_vec, embed) for embed in embed_dic]
        )
        idx = np.argsort(similarities)[::-1]
        values = np.sort(similarities)[::-1]
        return idx[:topk], values[:topk]

    def cos_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main(args: argparse.Namespace) -> None:
    texts, doc_paths = get_data(return_path=True)
    stopwords = get_stopword()
    preprocessor = Preprocessor(stopwords=stopwords, include_pos=args.include_pos)
    sentences = preprocessor.create_dataet(texts)
    embedder = Embedder(sentences, args.filename)
    embed = embedder.create_embed(sentences)

    idx, value = embedder.get_similar_sentence(["ビジネスマン"], embed)
    for i in range(len(idx)):
        print(idx[i], value[i], doc_paths[idx[i]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="an example program")
    parser.add_argument(
        "--filename", default="mymodel.model", type=str, help="filename of model"
    )
    parser.add_argument(
        "--include_pos",
        default=["名詞", "動詞", "形容詞"],
        nargs="*",
        type=str,
        help="list of pos not to be removed",
    )
    args = parser.parse_args()
    main(args)
