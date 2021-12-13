#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import click

from utils import tell_us_youre_running, tell_us_return_size


@tell_us_youre_running
@tell_us_return_size
def get_release_group_tags(mbdump_path):
    genres = get_genres(mbdump_path)
    header_list = ["release_group_id", "tag_id", "upvotes", "updated_at"]

    release_group_tags = pd.read_csv(
        f"{mbdump_path}/release_group_tag", sep="\t", names=header_list
    )
    return release_group_tags.merge(
        genres, left_on="tag_id", right_on="id", suffixes=("_release_group", "_genre")
    )


@tell_us_youre_running
@tell_us_return_size
def get_release_groups(mbdump_path):
    header_list = ["id", "mbid", "name", "artist_id", "a", "b", "c", "updated_at"]

    return pd.read_csv(f"{mbdump_path}/release_group", sep="\t", names=header_list)


@tell_us_youre_running
@tell_us_return_size
def get_tags(mbdump_path):
    header_list = ["id", "name", "ref_count"]

    return pd.read_csv(f"{mbdump_path}/tag", sep="\t", names=header_list)


@tell_us_youre_running
@tell_us_return_size
def get_genres(mbdump_path):
    tags = get_tags(mbdump_path)
    header_list = ["id", "mbid", "name", "a", "b", "updated_at"]

    genre_names = pd.read_csv(f"{mbdump_path}/genre", sep="\t", names=header_list)

    return tags[tags["name"].isin(genre_names["name"])]


@tell_us_youre_running
@tell_us_return_size
def get_artists(mbdump_path):
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
        f"{mbdump_path}/artist",
        sep="\t",
        names=header_list,
        dtype={"disambiguation": str},
    )
    return artists[["id", "name", "mbid"]]


@tell_us_youre_running
@tell_us_return_size
def get_albums(mbdump_path):
    release_groups = get_release_groups(mbdump_path)
    artists = get_artists(mbdump_path)
    return release_groups.merge(
        artists,
        left_on="artist_id",
        right_on="id",
        suffixes=("_release_group", "_artist"),
    )


def describe(r):
    return f"Album: {r['name_release_group']}\nArtist: {r['name_artist']}"


@tell_us_youre_running
@tell_us_return_size
def get_album_genres(mbdump_path):
    release_group_tags = get_release_group_tags(mbdump_path)

    albums = get_albums(mbdump_path)
    album_genres = albums.merge(
        release_group_tags,
        left_on="id_release_group",
        right_on="release_group_id",
        suffixes=("", "_genre"),
    )
    return album_genres.rename({"name": "genre"}, axis=1)[
        ["release_group_id", "name_release_group", "name_artist", "genre"]
    ]


@click.command()
@click.argument("mbdump_path", type=click.Path(exists=True, file_okay=False))
def simplify_mbdump(mbdump_path):
    album_genres = get_album_genres(mbdump_path)
    album_genres.to_csv(f"{mbdump_path}/album_genres.csv")
    print("Saved")


if __name__ == "__main__":
    simplify_mbdump()
