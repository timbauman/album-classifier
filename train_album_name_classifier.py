#!/usr/bin/env python
# coding: utf-8

from fastai.text.all import *

import pandas as pd
import click
import time

RANDOM_SEED = 12345  # not random at all. we want the sampling to be deterministic

# helper
def tell_us_youre_running(func):
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}")
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} ran in {time.time() - start_time} seconds")
        return res

    return wrapper


@tell_us_youre_running
def get_release_group_tags():
    genres = get_genres()
    header_list = ["release_group_id", "tag_id", "upvotes", "updated_at"]

    release_group_tags = pd.read_csv(
        "/Users/timothybauman/mbdump/release_group_tag", sep="\t", names=header_list
    )
    return release_group_tags.merge(
        genres, left_on="tag_id", right_on="id", suffixes=("_release_group", "_genre")
    )


@tell_us_youre_running
def get_release_groups():
    header_list = ["id", "mbid", "name", "artist_id", "a", "b", "c", "updated_at"]

    return pd.read_csv(
        "/Users/timothybauman/mbdump/release_group", sep="\t", names=header_list
    )
    # return release_group[['id', 'name', 'artist_id', 'mbid']]


@tell_us_youre_running
def get_tags():
    header_list = ["id", "name", "ref_count"]

    return pd.read_csv("/Users/timothybauman/mbdump/tag", sep="\t", names=header_list)
    # return tags[['id', 'name']]


@tell_us_youre_running
def get_genres():
    tags = get_tags()
    header_list = ["id", "mbid", "name", "a", "b", "updated_at"]

    genre_names = pd.read_csv(
        "/Users/timothybauman/mbdump/genre", sep="\t", names=header_list
    )

    return tags[tags["name"].isin(genre_names["name"])]


@tell_us_youre_running
def get_artists():
    header_list = [
        "id",
        "mbid",
        "name",
        "sort_name",
        "begin",
        "b",
        "c",
        "end",
        "d",
        "e",
        "gender_id",
        "area_id",
        "h",
        "disambiguation",
        "j",
        "updated_at",
        "k",
        "l",
        "m",
    ]

    artists = pd.read_csv(
        "/Users/timothybauman/mbdump/artist",
        sep="\t",
        names=header_list,
        dtype={"disambiguation": str},
    )
    return artists[["id", "name", "mbid"]]


@tell_us_youre_running
def get_albums():
    release_groups = get_release_groups()
    artists = get_artists()
    return release_groups.merge(
        artists,
        left_on="artist_id",
        right_on="id",
        suffixes=("_release_group", "_artist"),
    )


@tell_us_youre_running
def get_top_tags(release_group_tags, num_genres):
    tag_counts = (
        release_group_tags.groupby("name")
        .count()
        .sort_values("release_group_id", ascending=False)
    )
    return tag_counts.head(num_genres)


def describe(r):
    return f"Album: {r['name_release_group']}\nArtist: {r['name_artist']}"


@tell_us_youre_running
def get_album_genres(num_genres):
    release_group_tags = get_release_group_tags()
    top_tags = get_top_tags(release_group_tags, num_genres)
    filtered_release_group_tags = release_group_tags[
        release_group_tags["name"].isin(top_tags.index)
    ]

    albums_to_genre_set = filtered_release_group_tags.groupby(
        filtered_release_group_tags["release_group_id"]
    ).aggregate(
        {
            "tag_id": lambda x: set(x),
            "name": lambda x: set(x),
        }
    )

    albums = get_albums()
    album_genres = albums.merge(
        albums_to_genre_set,
        left_on="id_release_group",
        right_on="release_group_id",
        suffixes=("", "_genre"),
    )
    album_genres["description"] = album_genres.apply(describe, axis=1)
    return album_genres


@tell_us_youre_running
def sample_album_genres(album_genres, num_samples):
    return (
        album_genres.sample(n=num_samples, random_state=RANDOM_SEED)
        if num_samples is not None
        else album_genres
    )


@tell_us_youre_running
def build_learner(album_genres):
    def get_y(r):
        return r["name"]

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
    learner.fine_tune(num_epochs, lr=2e-3)
    learner.save(model_filename)


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
    album_genres = get_album_genres(num_genres)
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
