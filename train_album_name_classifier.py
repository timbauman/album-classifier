#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import click

from fastai.text.all import *

from utils import tell_us_youre_running
from torch.profiler import profile, record_function, ProfilerActivity

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


def find_least_common_genre(tags, genres):
    genre_set = set(genres)
    return next(tag for tag in tags if tag in genre_set)


@tell_us_youre_running
def get_top_album_genres(num_genres, multi_category):
    album_genres = pd.read_csv("/Users/timothybauman/mbdump/album_genres.csv")
    top_tags = get_top_tags(album_genres, num_genres)
    filtered_album_genres = album_genres[album_genres["genre"].isin(top_tags.index)]
    filtered_album_genres_set = filtered_album_genres.groupby(
        filtered_album_genres["release_group_id"]
    ).aggregate(
        {
            "genre": (lambda x: set(x))
            if multi_category
            else partial(find_least_common_genre, list(reversed(top_tags.index))),
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


def get_y(r):
    return r["genre"]


@tell_us_youre_running
def build_learner(album_genres, multi_category):

    dblock = DataBlock(
        get_x=describe,
        get_y=get_y,
        blocks=(
            TextBlock.from_df("description"),
            MultiCategoryBlock if multi_category else CategoryBlock,
        ),
    )
    dls = dblock.dataloaders(album_genres)

    return text_classifier_learner(
        dls, AWD_LSTM, metrics=(accuracy_multi if multi_category else accuracy)
    ).to_fp16()


@tell_us_youre_running
def train_learner(learner, model_filename, num_epochs, base_lr):
    learner.fine_tune(num_epochs, base_lr=base_lr)
    learner.export(model_filename)


@tell_us_youre_running
def test_learner(learner, album_genres):
    foo = album_genres.sample(n=30)
    foo["result"] = foo["description"].map(learner.predict)
    print(foo)


def train_album_name_classifier_impl(
    album_genres, num_samples, num_epochs, multi_category, model_filename, base_lr
):
    album_genres_sample = sample_album_genres(album_genres, num_samples)
    learner = build_learner(album_genres_sample, multi_category)
    train_learner(learner, model_filename, num_epochs, base_lr)
    return learner


@click.command()
@click.option("--num_genres", default=2)
@click.option("--num_samples", default=100000)
@click.option("--num_epochs", default=4)
@click.option("--base_lr", default=2e-3)  # use find_lr to find appropriate numbers
@click.option("--multi-category/--single-category", default=True)
@click.argument(
    "model_filename", type=click.STRING
)  # not using a path because it's a relative path
def train_album_name_classifier(
    num_genres, num_samples, num_epochs, base_lr, multi_category, model_filename
):
    album_genres = get_top_album_genres(num_genres, multi_category)
    learner = train_album_name_classifier_impl(
        album_genres, num_samples, num_epochs, multi_category, model_filename, base_lr
    )
    test_learner(learner, album_genres)


if __name__ == "__main__":
    train_album_name_classifier()
