# Module 3 — Pearl millet disease detection

Classify a pearl-millet leaf from its colour. The script resizes each image to 64×64 and keeps 30 numbers: the mean and spread of red, green, and blue, plus a small histogram of each channel. A decision tree, logistic regression, a random forest, and a linear SVM are compared with a majority-class guess.

There is no neural network in this folder. PyTorch is a large install, and the full Zenodo image set is about 600 MB. Both were left out so this module runs on a normal PC.

## Images included

Zenodo record [10.5281/zenodo.17086194](https://zenodo.org/records/17086194), licence CC BY 4.0. Only the three small archives are used:

| Folder | Class | Images |
| --- | --- | --- |
| `healthy` | Healthy leaves | 41 |
| `sclerospora_graminicola` | *Sclerospora graminicola* | 30 |
| `exserohilum_rostratum` | *Exserohilum rostratum* | 9 |

Not included, because of size: *Moesziomyces bullatus* (about 230 MB) and *Pyricularia* (about 370 MB). The Roboflow copy of pearl-millet images is not used either.

`download_data.py` fetches those three archives, unpacks them into `data/images/<class>/`, and deletes the zip.

## Run

```bash
pip install -r requirements.txt
python train_baselines.py
```

Split: stratified 80/20, seed 42. That is 64 training images and 16 test images. No augmentation. The script writes filenames of a few mistakes to `results/misclassified_examples.csv` and does not copy the photo files.

## Results on these 80 images

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Majority class | 0.500 | 0.167 | 0.333 | 0.222 |
| Decision tree | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic regression | 0.938 | 0.963 | 0.833 | 0.869 |
| Linear SVM | 0.938 | 0.963 | 0.833 | 0.869 |
| Random forest | 0.938 | 0.952 | 0.833 | 0.863 |

The test set has 8 healthy, 6 downy-mildew, and 2 leaf-blight images. A perfect tree score on 16 images is too small to trust. The forest misses one of the two *Exserohilum* test images (recall 0.50). Read macro F1 and `results/confusion_matrix_best.png`.

## Limitations

- Two of the five Zenodo diseases are absent. This is a size choice, not a claim that those diseases are unimportant.
- Photos are from research plots in Burkina Faso, not from the project hardware.
- Folder names do not identify the plant, so similar photos of one leaf can land in both train and test.
- Colour numbers miss lesion shape. They also react to lighting and background.

## What proceeds

Keep this colour-feature script as the light baseline. Add the two large disease classes only when there is disk space for them, and only as a separate download. A convolutional model is a later step, after this comparison has been read.
