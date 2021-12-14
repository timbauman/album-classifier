#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import click

from fastai.text.all import *

from utils import tell_us_youre_running

RANDOM_SEED = 12345  # not random at all. we want the sampling to be deterministic


def describe(r):
    return f"Album: {r['name_release_group']}\nArtist: {r['name_artist']}"


@tell_us_youre_running
def get_top_tags(album_genres, num_genres):
    tag_counts = (
        album_genres.groupby("genre")
        .count()
        .sort_values("name_release_group", ascending=False)
    )
    return tag_counts.head(num_genres)


@tell_us_youre_running
def get_top_album_genres(num_genres):
    album_genres = pd.read_csv("/Users/timothybauman/mbdump/album_genres.csv")
    top_tags = get_top_tags(album_genres, num_genres)
    filtered_album_genres = album_genres[album_genres["genre"].isin(top_tags.index)]
    filtered_album_genres
    filtered_album_genres_set = filtered_album_genres.groupby(
        filtered_album_genres["release_group_id"]
    ).aggregate(
        {
            "genre": lambda x: set(x),
            "name_release_group": "first",
            "name_artist": "first",
        }
    )
    filtered_album_genres_set["description"] = filtered_album_genres_set.apply(
        describe, axis=1
    )
    return filtered_album_genres_set


@tell_us_youre_running
def sample_album_genres(album_genres, num_samples):
    return (
        album_genres.sample(n=num_samples, random_state=RANDOM_SEED)
        if num_samples > 0
        else album_genres
    )


@tell_us_youre_running
def build_learner(album_genres):
    def get_y(r):
        return r["genre"]

    dblock = DataBlock(
        get_x=describe,
        get_y=get_y,
        blocks=(TextBlock.from_df("description"), MultiCategoryBlock),
    )
    dls = dblock.dataloaders(album_genres)

    return text_classifier_learner(dls, AWD_LSTM, metrics=accuracy_multi).to_fp16()


# preds,targs = learner.get_preds()


# xs = torch.linspace(0.05,0.95,29)
# accs = [accuracy_multi (preds, targs, thresh=i, sigmoid=False) for i in xs]
# plt.plot(xs,accs)


@tell_us_youre_running
def train_learner(learner, model_filename, num_epochs):
    learner.fine_tune(num_epochs, base_lr=2e-3)
    learner.export(model_filename)


@tell_us_youre_running
def test_learner(learner, album_genres):
    foo = album_genres.sample(n=30)
    foo["result"] = foo["description"].map(learner.predict)
    print(foo)


@click.command()
@click.option("--num_genres", default=2)
@click.option("--num_samples", default=100000)
@click.option("--num_epochs", default=4)
@click.argument(
    "model_filename", type=click.STRING
)  # not using a path because it's a relative path
def train_album_name_classifier(num_genres, num_samples, num_epochs, model_filename):
    album_genres = get_top_album_genres(num_genres)
    album_genres_sample = sample_album_genres(album_genres, num_samples)
    learner = build_learner(album_genres_sample)
    train_learner(learner, model_filename, num_epochs)
    test_learner(learner, album_genres)


if __name__ == "__main__":
    train_album_name_classifier()

# learn2 = learner.load('10genres-4epochs')


# preds,targs = learn2.get_preds()
# xs = torch.linspace(0.05,0.95,29)
# accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
# plt.plot(xs,accs);


# def describe(r): return f"Album name: {r['name_release_group']}\nArtist: {r['name_artist']}"
