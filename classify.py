#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import click


from fastai.text.all import *

from train_album_name_classifier import *


@click.command()
@click.argument("model_filename", type=click.Path(exists=True, dir_okay=False))
@click.argument("artist_name", type=click.STRING)
@click.argument("album_name", type=click.STRING)
def classify(model_filename, artist_name, album_name):
    learner = load_learner(model_filename)
    print(
        learner.predict(
            describe({"name_release_group": album_name, "name_artist": artist_name})
        )
    )


if __name__ == "__main__":
    classify()
