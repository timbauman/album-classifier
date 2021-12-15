# Simple NN training to classify albums by genre

This is a simple classifier built with fastai. It isn't super intelligent and probably overfits, but it's kind of fun to mess around with.

1. Install [Conda](https://docs.conda.io/en/latest/) and create a conda environment for this repo:
```
conda env create --prefix ./env --file envname.yml
conda activate ./env
```

2. Download MusicBrainz's `mbdump` and `mbdump-derived` databases here: https://musicbrainz.org/doc/MusicBrainz_Database/Download and extract them into a single `mbdump` directory

3. Run the following to preprocess the data into album_genres.csv (which speeds up the training time):
```
$ python simplify_mbdump.py [path to mbdump directory]
```

4. Run the following to train a model:
```
$ python train_album_name_classifier.py --single-category --num_genres 2 --num_samples 100000 --num_epochs 5 ~/mbdump [model-filename]
```

5. Run the following to classify an album:
```
$ python classify.py [model-filename] [artist-name] [album-name]
('rock', TensorText(167), TensorText([4.1902e-06, 1.1308e-06, 1.1814e-05, 4.5779e-02, 1.6902e-04, 7.6783e-06,                    
        9.8771e-05, 7.6575e-03, 2.3041e-02, 6.6032e-06, 8.2277e-04, 1.2604e-03,
        2.9098e-06, 9.2480e-06, 2.6381e-05, 8.9736e-05, 1.6072e-05, 2.2026e-03,
        1.1345e-02, 1.6803e-05, 9.5393e-06, 3.9641e-06, 1.3059e-05, 1.9589e-04,
        2.6256e-06, 3.3159e-05, 3.4382e-05, 3.3456e-05, 5.9334e-02, 1.3608e-04,
        9.3923e-04, 2.2192e-04, 2.3283e-05, 6.6489e-06, 1.0442e-04, 1.2206e-06,
        2.5483e-03, 9.4070e-05, 3.7537e-05, 3.4574e-06, 1.9097e-04, 1.7347e-05,
        1.1202e-04, 1.9750e-05, 1.6865e-04, 2.4176e-05, 4.4516e-05, 2.1436e-04,
        2.8335e-05, 4.4480e-05, 3.3900e-06, 1.0803e-04, 3.8702e-06, 1.1781e-05,
        9.0424e-05, 9.4428e-06, 5.9107e-06, 1.7255e-04, 4.2235e-04, 2.2428e-05,
        9.5529e-04, 9.5483e-06, 7.5577e-06, 2.2363e-05, 6.2487e-06, 4.9185e-06,
        5.9488e-04, 7.8913e-04, 2.5435e-04, 2.0173e-03, 1.7159e-02, 4.7015e-04,
        9.5931e-06, 2.9756e-03, 1.3548e-04, 7.1564e-04, 1.6295e-05, 1.2318e-06,
        1.5316e-06, 2.9013e-03, 2.5590e-03, 7.0527e-06, 6.4352e-07, 1.1121e-04,
        1.1051e-05, 6.3224e-07, 1.5949e-06, 1.7473e-05, 9.1473e-05, 3.3164e-06,
        2.3738e-06, 3.9434e-06, 3.2229e-03, 1.3697e-06, 1.2301e-05, 1.0087e-06,
        3.5786e-04, 2.2072e-03, 2.0283e-04, 3.7670e-05, 4.2059e-05, 1.4318e-03,
        1.2390e-04, 3.3500e-04, 1.3517e-03, 3.4900e-05, 8.8499e-06, 8.5159e-04,
        4.3092e-06, 9.3898e-05, 3.4970e-04, 3.6927e-03, 1.0016e-03, 8.5301e-06,
        8.1141e-04, 1.1417e-03, 4.3840e-05, 7.0342e-05, 1.7356e-04, 1.2472e-04,
        1.0344e-05, 5.5772e-07, 8.1559e-07, 1.4173e-04, 1.7382e-06, 1.3131e-04,
        6.3830e-05, 2.5752e-05, 1.0743e-06, 4.6273e-05, 1.5613e-05, 4.1313e-06,
        1.1672e-04, 6.0951e-03, 3.4711e-05, 1.0513e-04, 2.8324e-03, 9.2820e-05,
        2.8381e-06, 1.6370e-05, 2.1984e-06, 5.7339e-02, 1.8820e-04, 3.0003e-04,
        2.3830e-01, 8.5557e-06, 2.3152e-05, 2.6347e-05, 1.0089e-03, 2.8854e-05,
        2.1878e-06, 1.1054e-06, 2.1192e-03, 1.5939e-05, 3.9388e-05, 5.1581e-05,
        2.5409e-03, 9.6534e-06, 5.7227e-04, 7.9071e-02, 3.6422e-06, 6.5799e-07,
        2.1203e-03, 1.4654e-03, 6.0516e-04, 7.7403e-04, 1.9472e-06, 2.7237e-01,
        1.0030e-01, 9.6627e-05, 8.0954e-05, 2.9705e-04, 4.5190e-05, 6.6958e-05,
        7.2174e-04, 6.0310e-06, 1.5149e-04, 1.8229e-02, 5.4225e-04, 2.7219e-05,
        1.4981e-05, 4.3340e-04, 9.1035e-06, 1.7907e-04, 4.8153e-06, 1.3493e-04,
        5.5937e-06, 4.0855e-06, 2.4779e-04, 1.5441e-06, 1.2930e-03, 5.5667e-05,
        1.6411e-04, 1.5742e-04, 2.7108e-06, 1.8835e-05, 3.6169e-06, 2.4086e-04,
        9.1328e-07, 8.4468e-05]))
```

It prints the raw output since the classifier might have different outputs depending on the number of genres and multi vs single classification
