# Image Analysis and Processing - Second Assignment

_Konstantinos Konstantinidis_  
_Student number: 7115152400017_

## Data

The data used were downloaded from Kaggle, [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). They consist of several hundreds of X-ray images taken from patients, labeled as "Normal" and "Pneumonia". The data are pre-split to __train__ ($5218$ images), __test__ ($624$ images), and __validation__ ($18$ images) sets. The dataset is heavily imbalanced, with $1585$ "Normal" images and $4275$ "Pneumonia" images. The images are grayscale, and they have varying sizes. Based on the __average intensity distribution__ of the two classes, we can see that the "Pneumonia" class has a higher `std` on the 150 to 200 range of intensities, which is an interesting finding, indicating that the intensities could be a valuable feature.

To conclude, the data must be merged because the __train-test-val__ split is suboptimal and the class imbalance will also have to be addressed, by creating synthetic images by augmenting the existing ones.

All relevant visualizations are in the [eda.ipynb](notebooks/eda.ipynb) notebook.

## Preprocessing

Preprocessing will consist of augmenting the images of the underrepresented class in order to generate new ones to balance the dataset. Also an image can be augmented with a probability of $0.5$, in order to introduce some more variance in the dataset.

An augmentations can be one of the following:

- Random rotation between -15 and 15 degrees
- Horizontal flip
- Brightness adhustment by a factor between 0.8 and 1.2.
- Contrast adjustment by a factor between 0.8 and 1.2.
- Zoom (in or out, crop or pad) by a factor between 0.8 and 1.2.
- Random shift (translation) by a factor between -20 and 20 in the x and y axis.
- Addition of salt and pepper noise (black and white pixels) taking up 2.5% of all pixels.

The implementation is in the [preprocessing.ipynb](notebooks/preprocessing.ipynb) notebook.

## Feature extraction and Classical ML

### Feature extraction

To train a classical model, instead of passing all the pixels as feratures, which would be impractical, a better approach is to extract features from the images.

The extracted features are:

- __General statistics__
  - `mean`
  - `median`
  - `std`
  - `entropy`
  - `energy`
- __Histogram statistics__Using bins with size 50 (i.e. intensities [0-50], [51-100], ..., [201-250]), take:
  - `mean`
  - `std`
- __Otsu's threshold__The intensity value that best separates the pixels in two classe, determined by minimizing intra-class variance.
- __Number of edges__The number of edges in the image, based on two user defined thresholds.
- __Number of horizontal and vertical lines__Using the edges of the images, and again specified using user defined thresholds.
- __Fast Furier Transofrm (FFT) features__Computes the FFT of the image and based on the magnitute spectrum extracts:
  - `mean`
  - `std`
- __Fractal dimension__Calculates the fractal dimension of the image using the box-counting method.
- __Hu's invariant moments__
  These are seven moments extracted from the image that are invariant with respect to translation, scale, and rotation.

The extracted features along with the label of the corresponding image is stored in a `.csv` file.

The implementation is in the [feature_extraction.ipynb](notebooks/feature_extraction.ipynb) notebook.

### Classical ML

An __SVM__ classifier was trained, using `mRMR` for feature selection and `Optuna` for hyperparameter tuning.

The selected features were:

- `n_edges`
- `vertical_lines`
- `hist_std_151_200`
- `horizontal_lines`
- `fft_mean`
- `hist_std_0_50`
- `hist_means_151_200`
- `hist_std_101_150`
- `std`
- `hu_moment_4`

After training, the model was tested against a hold-out set, yielding very good performance, having a __Matthews Correlation Coefficient__ of $0.697$, __f1 Score__ of $0.848$, and __Accuracy__ of $0.848$, indicating that this model, using only these 10 features can accurately classify chest X-ray images as "Normal" or "Pneumonia".

The implementation is in the [classical_ml.ipynb](notebooks/classical_ml.ipynb) notebook.

## Deep learning approach

To classify these images, I opted for the `ResNet50` model, with its pretraining weights and by only training the last layer. After training, the model was tested against a hold-out set.

The final model is an extremely strong classifier with Classification Metrics:

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 0.9344 |
| MCC         | 0.8696 |
| F1-Score    | 0.9332 |
| ROC-AUC     | 0.9822 |
| Sensitivity | 0.9136 |
| Specificity | 0.9554 |

The implementation is in the [nn_ml.ipynb](notebooks/nn_ml.ipynb) notebook.



## Notes

All functions are in the `src` directory and all implementations are in the `notebooks` directory. The models were saved in the `models` directory.

The data are in the `chest_xray`, `merged`, `preprocessed`, and `extracted_features` directories which are omitted from this repository in order to save space and keep it cleaner, so to recreate the results __read the following brief instructions:__

- Clone this repository and cd into it.
- Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- The file `requirements.txt` contains the packages used. To recreate the environment to re-run the notebooks, run:
  ```python
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  and select the appropriate environment in the notebooks.
- Run the notebooks in the order mentioned in this `.md`.
