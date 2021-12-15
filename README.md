# Simple NN training to classify albums by genre

This is a simple classifier built with fastai. It isn't super intelligent and probably overfits, but it's kind of fun to mess around with.

1. Install [Conda](https://docs.conda.io/en/latest/) and create a conda environment for this repo:
```
% conda env create --prefix ./env --file envname.yml
% conda activate ./env
```

2. Download MusicBrainz's `mbdump` and `mbdump-derived` databases here: https://musicbrainz.org/doc/MusicBrainz_Database/Download and extract them into a single `mbdump` directory

3. Run the following to preprocess the data into album_genres.csv (which speeds up the training time):
```
% python simplify_mbdump.py [path to mbdump directory]
```

4. Run the following to train a model:
```
% python train_album_name_classifier.py --single-category --num_genres 2 --num_samples 100000 --num_epochs 5 ~/mbdump [model-filename]
```

5. Run the following to classify an album:
```
% python classify.py [model-filename] [artist-name] [album-name]
```

For example:
```
% python classify.py ../200genres-10epochs-single "DJ Tim" "Tim Is A Genius (Dance Remixes)"
('house', TensorText(99), TensorText([7.0668e-03, 6.9381e-04, 4.3516e-05, 3.9551e-04, 5.2087e-04, 4.1603e-06, ...
```

It prints the raw output since the classifier might have different outputs depending on the number of genres and multi vs single classification
