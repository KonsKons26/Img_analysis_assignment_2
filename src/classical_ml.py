import os
import numpy as np
import pandas as pd
import cv2 as cv
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from mrmr import mrmr_classif
import optuna
import joblib

from src.utils import read_image


class FeatureExtractor:
    """
    A class for extracting features from images for classical machine learning
    models. The features include first-order statistics, histogram statistics,
    Otsu's threshold, edge counts, horizontal and vertical line counts,
    FFT mean and std, fractal dimension, and Hu's seven invariant moments.

    Attributes
    ----------
    FEATURE_NAMES : list
        A list of feature names corresponding to the extracted features.
    otsu_thr : int
        The threshold value for Otsu's method, default is 128.
    edge_thr1 : int
        The first threshold for Canny edge detection, default is 50.
    edge_thr2 : int
        The second threshold for Canny edge detection, default is 150.
    line_thr : int
        The threshold for Hough Line Transform to detect lines, default is 1.
    hor_thr : int
        The threshold for detecting horizontal lines, default is 1.
    ver_thr : int
        The threshold for detecting vertical lines, default is 1.
    frac_thr : int
        The threshold for the fractal dimension calculation, default is 128.

    Public Methods
    --------------
    parallel_extract_features(path: str) -> pd.DataFrame
        Extracts features from images in the specified path and returns them
        as a pandas DataFrame. The path should contain two subdirectories,
        "NORMAL" and "PNEUMONIA", each containing images of the respective
        class. The DataFrame will contain a label column (0 for NORMAL, 1 for
        PNEUMONIA) and columns for each extracted feature.

    Private Methods
    ---------------
    _extract_features_from_image(img, label)
        Extracts features from the image and returns them as a numpy array.
    _first_order_stats(img)
        Extracts first-order statistics from the image.
    _hist_stats(img)
        Extracts histogram statistics from the image.
    _otsu_intraclass_variance(img)
        Computes the Otsu's intraclass variance for the image.
    _count_edges(img)
        Counts the number of edges in the image using Canny edge detection.
    _count_horizontals_verticals(edges_img)
        Counts the number of horizontal and vertical lines in the image using
        Hough Line Transform.
    _fft_features(img)
        Computes the Fast Fourier Transform (FFT) of the image and extracts
        the mean and standard deviation of the magnitude spectrum.
    _fractal_dimension(img)
        Computes the fractal dimension of the image using the box-counting
        method.
    _hu_moments(img)
        Computes Hu's seven invariant moments from the image.
    """

    FEATURE_NAMES = [
        "mean", "median", "std", "entropy", "energy",
        
        "hist_means_0_50", "hist_means_51_100", "hist_means_101_150",
        "hist_means_151_200", "hist_means_201_250",
        "hist_std_0_50", "hist_std_51_100", "hist_std_101_150",
        "hist_std_151_200", "hist_std_201_250",

        "otsu_threshold",

        "n_edges",
        
        "horizontal_lines", "vertical_lines",

        "fft_mean", "fft_std",

        "fractal_dim",

        "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
        "hu_moment_5", "hu_moment_6", "hu_moment_7"
    ]

    def __init__(
            self,
            edge_thr1: int = 50,
            edge_thr2: int = 150,
            line_thr: int = 1,
            hor_thr: int = 1,
            ver_thr: int = 1,
            frac_thr: int = 128
    ):
        self.edge_thr1 = edge_thr1
        self.edge_thr2 = edge_thr2
        self.line_thr = line_thr
        self.hor_thr = hor_thr
        self.ver_thr = ver_thr
        self.frac_thr = frac_thr

    def parallel_extract_features(
            self,
            path: str,
            num_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Extracts features from images in the specified path and returns them
        as a pandas DataFrame. The path should contain two subdirectories,
        "NORMAL" and "PNEUMONIA", each containing images of the respective
        class. The DataFrame will contain a label column (0 for NORMAL, 1 for
        PNEUMONIA) and columns for each extracted feature.

        Parameters
        ----------
        path : str
            The base directory containing 'NORMAL' and 'PNEUMONIA'
            subdirectories.
        num_jobs : int, optional
            The number of parallel jobs to run. -1 means using all available CPU
            cores. Default is -1.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row represents an image and columns are its
            extracted features and the 'label' (0 for normal, 1 for pneumonia).
        """
        normal_dir = os.path.join(path, "NORMAL")
        pneumonia_dir = os.path.join(path, "PNEUMONIA")

        image_paths_with_labels = []

        # Collect NORMAL images
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                ):
                    image_paths_with_labels.append(
                        (os.path.join(normal_dir, img_name), 0)
                    )
        else:
            print(f"Warning: NORMAL directory not found at {normal_dir}")

        # Collect PNEUMONIA images
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                ):
                    image_paths_with_labels.append(
                        (os.path.join(pneumonia_dir, img_name), 1)
                    )
        else:
            print(f"Warning: PNEUMONIA directory not found at {pneumonia_dir}")

        # Check if any images were found
        if not image_paths_with_labels:
            print("No images found in the specified directories.")
            return pd.DataFrame()

        print(f"Found {len(image_paths_with_labels)} images to process.")

        def _process_single_image_for_parallel(image_path: str, label: int):
            """Process a single image to extract features."""
            try:
                img = read_image(image_path)
                features_array = self._extract_features_from_image(
                    img=img, label=label
                )
                return features_array
            except FileNotFoundError as fnf_e:
                print(f"File error processing {image_path}: {fnf_e}")
                return None
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        # Use joblib to parallelize the feature extraction
        results = Parallel(n_jobs=num_jobs)(
            delayed(_process_single_image_for_parallel)(img_path, label)
            for img_path, label in tqdm(
                image_paths_with_labels,
                desc="Extracting Features", unit="images"
            )
        )

        # Filter out None results (failed extractions)
        valid_results = [res for res in results if res is not None]

        # Check if any features were successfully extracted
        if not valid_results:
            print("No features were successfully extracted.")
            return pd.DataFrame(columns=["label"] + self.FEATURE_NAMES)

        # Convert the list of valid results to a numpy array
        all_features = np.array(valid_results)

        # Create a DataFrame with the features and labels
        df = pd.DataFrame(all_features, columns=["label"] + self.FEATURE_NAMES)
        df["label"] = df["label"].astype(int)

        return df


    def _extract_features_from_image(
            self,
            img: np.ndarray,
            label: int
        ) -> np.ndarray:
        """
        Extracts features from the image and returns them as a numpy array.
        The features include first-order statistics, histogram statistics,
        Otsu's threshold, edge counts, horizontal and vertical line counts,
        FFT mean and std, fractal dimension, and Hu's seven invariant moments.

        Returns
        -------
        np.ndarray
            A numpy array containing the label and the extracted features.
        """

        # General stats
        mean, median, std, entropy, energy = self._first_order_stats(img)

        # Histogram stats
        hist_means, hist_stds = self._hist_stats(img)

        # Otsu's threshold
        otsus_threshold = self._otsu_intraclass_variance(img)

        # Number of edges
        edges_count, edges_img = self._count_edges(img)

        # Horizontal and vertical lines
        hor, ver = self._count_horizontals_verticals(edges_img)

        # FFT mean and std
        fft_mean, fft_std = self._fft_features(img)

        # Fractal dim
        fractal_dim = self._fractal_dimension(img)

        # Hu's seven invariant moments
        hu_moments = self._hu_moments(img)

        return np.array([
            label,

            mean, median, std, entropy, energy,

            hist_means[0], hist_means[1], hist_means[2], hist_means[3],
            hist_means[4], hist_stds[0], hist_stds[1], hist_stds[2],
            hist_stds[3], hist_stds[4],

            otsus_threshold,

            edges_count,

            hor, ver,

            fft_mean, fft_std,

            fractal_dim,

            hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
            hu_moments[4], hu_moments[5], hu_moments[6]
        ])

    def _first_order_stats(self, img):
        """Extracts first-order statistics from the image."""
        return [
            np.mean(img),
            np.median(img),
            np.std(img),
            -np.sum(img * np.log2(img + 1e-10)),
            np.sum(img**2)
        ]

    def _hist_stats(self, img):
        """Extracts histogram statistics from the image."""    
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        sum_hist = np.sum(hist)
        normalized_hist = hist.flatten(
        ) / sum_hist if sum_hist > 0 else np.zeros(256)

        hist_means = [
            np.mean(normalized_hist[i: i+50]) for i in range(0, 250, 50)
        ]
        hist_stds = [
            np.std(normalized_hist[i: i+50]) for i in range(0, 250, 50)
        ]

        return hist_means, hist_stds

    def _otsu_intraclass_variance(self, img):
        """Computes the Otsu's threshold for the image."""
        if img.size == 0:
            return 0.0

        if img.dtype != np.uint8:
            img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(
                np.uint8
            )
        else:
            img_8bit = img

        ret, _ = cv.threshold(
            img_8bit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        actual_otsu_threshold = ret

        pixels_above_thr = img[img >= actual_otsu_threshold]
        pixels_below_thr = img[img < actual_otsu_threshold]

        variance_sum = 0.0
        if pixels_above_thr.size > 0:
            if pixels_above_thr.size > 1:
                variance_sum += np.mean(pixels_above_thr) * np.var(
                    pixels_above_thr
                )
            else:
                variance_sum += np.mean(pixels_above_thr) * 0.0

        if pixels_below_thr.size > 0:
            if pixels_below_thr.size > 1:
                variance_sum += np.mean(pixels_below_thr) * np.var(
                    pixels_below_thr
                )
            else:
                variance_sum += np.mean(pixels_below_thr) * 0.0

        return actual_otsu_threshold

    def _count_edges(self, img):
        """Counts the number of edges in the image using Canny edge
        detection."""
        img_copy = img.copy()
        if img_copy.dtype != np.uint8:
            img_copy = cv.normalize(
                img_copy, None, 0, 255, cv.NORM_MINMAX
            ).astype(np.uint8)
        img_copy = cv.GaussianBlur(img_copy, (7, 7), 0)
        edges = cv.Canny(img_copy, self.edge_thr1, self.edge_thr2)
        return np.sum(edges > 0), edges

    def _count_horizontals_verticals(self, edges_img):
        """Counts the number of horizontal and vertical lines in the image using
        Hough Line Transform."""
        hor, ver = 0, 0
        if len(edges_img.shape) == 3:
            edges_img = cv.cvtColor(edges_img, cv.COLOR_BGR2GRAY)

        lines = cv.HoughLines(edges_img, 1, np.pi/180, self.line_thr)
        if lines is not None:
            for line in lines:
                theta = line[0][1]
                angle = theta * 180 / np.pi

                if (
                    (90 - self.hor_thr < angle < 90 + self.hor_thr) or
                    (270 - self.hor_thr < angle < 270 + self.hor_thr)
                ):
                    hor += 1

                if (
                    (0 - self.ver_thr < angle < 0 + self.ver_thr) or
                    (180 - self.ver_thr < angle < 180 + self.ver_thr)
                ):
                    ver += 1

        return hor, ver

    def _fft_features(self, img):
        """Computes the Fast Fourier Transform (FFT) of the image and extracts
        the mean and standard deviation of the magnitude spectrum."""
        img_gray = img.copy()
        if len(img_gray.shape) == 3:
            img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
        fft = np.fft.fft2(img_gray)
        magnitude = np.abs(fft)
        return [np.mean(magnitude), np.std(magnitude)]

    def _fractal_dimension(self, img):
        """Computes the fractal dimension of the image using the box-counting
        method."""
        def boxcount(image, k):
            binary_image = (image > 0).astype(int)
            h, w = binary_image.shape
            h_pad = k - (h % k) if h % k != 0 else 0
            w_pad = k - (w % k) if w % k != 0 else 0
            padded_image = np.pad(
                binary_image,
                ((0, h_pad), (0, w_pad)),
                "constant",
                constant_values=0
            )
            S = np.add.reduceat(
                np.add.reduceat(
                    padded_image, np.arange(
                        0, padded_image.shape[0], k
                    ), axis=0
                ),
                np.arange(0, padded_image.shape[1], k), axis=1
            )
            return len(np.where(S > 0)[0]) * len(np.where(S > 0)[1])
        
        img_gray = img.copy()
        if len(img_gray.shape) == 3:
            img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)

        binary = (img_gray > self.frac_thr).astype(int)
        min_dim = min(binary.shape)
        sizes = [s for s in 2**np.arange(1, int(np.log2(min_dim)) + 1) if s > 0]

        if not sizes:
            return 0.0

        counts = [boxcount(binary, size) for size in sizes]
        valid_indices = np.array(counts) > 0
        if np.sum(valid_indices) < 2:
            return 0.0

        log_sizes = np.log(np.array(sizes)[valid_indices])
        log_counts = np.log(np.array(counts)[valid_indices])
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return -coeffs[0]

    def _hu_moments(self, img):
        """Computes Hu's seven invariant moments from the image."""
        img_gray = img.copy()
        if len(img_gray.shape) == 3:
            img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
        moments = cv.moments(img_gray)
        hu_moments = cv.HuMoments(moments).flatten()
        return list(hu_moments)

class MySVC:

    def __init__(
            self,
            model_dir: str,
            data: pd.DataFrame,
            target: str = "label",
            optuna_trials: int = 50,
            cv_folds: int = 5,
            max_features: int = 10,
            random_state: int = 42
        ):

        self.model_dir = model_dir

        self.data = data
        self.target = target
        self.X = self.data.copy()
        self.y = self.X.pop(self.target)
        self.optuna_trials = optuna_trials
        self.cv_folds = cv_folds
        self.max_features = max_features
        self.random_state = random_state

        self.best_model = None
        self.best_params = None
        self.best_mcc = None
        self.selected_features = None

    def _mrmr_feature_selection(self):
        K = min(self.max_features, self.X.shape[1])
        selected_features = mrmr_classif(
            X=self.X, y=self.y, K=K, show_progress=False
        )
        return selected_features

    def _objective(self, trial: optuna.Trial) -> float:
        selected_features = self._mrmr_feature_selection()
        X_selected = self.X[selected_features]

        svc_c = trial.suggest_float("C", 1e-5, 1e5, log=True)
        svc_kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid"])

        svc_gamma = "scale"
        if svc_kernel in ["rbf", "poly", "sigmoid"]:
            svc_gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        mcc_scores = []
        for fold, (train_index, val_index) in enumerate(
            skf.split(X_selected, self.y)
        ):
            X_train = X_selected.iloc[train_index]
            X_val = X_selected.iloc[val_index]
            y_train = self.y.iloc[train_index]
            y_val = self.y.iloc[val_index]

            model = SVC(
                C=svc_c,
                kernel=svc_kernel,
                gamma=svc_gamma,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mcc = matthews_corrcoef(y_val, y_pred)
            mcc_scores.append(mcc)

        avg_mcc = sum(mcc_scores) / len(mcc_scores)

        return avg_mcc

    def find_best_hyperparameters(self):
        print("Starting Optuna optimization...")
        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            self._objective,
            n_trials=self.optuna_trials, show_progress_bar=True
        )

        self.best_params = study.best_params
        self.best_mcc = study.best_value

        print(f"\nBest trial finished with MCC: {self.best_mcc}")
        print("Best hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        print("Performing final feature selection on full training set...")
        self.selected_features = self._mrmr_feature_selection()
        self.X_selected = self.X[self.selected_features]
        print(
            f"Selected {len(self.selected_features)} "
            f"features: {self.selected_features}"
        )

        self.best_model = SVC(
            random_state=self.random_state, **self.best_params
        )
        self.best_model.fit(self.X_selected, self.y)
        print("Best model trained successfully.")

        # Save model
        model_path = os.path.join(self.model_dir, "SVC_best_model.joblib")
        joblib.dump({
            "model": self.best_model,
            "selected_features": self.selected_features,
            "best_params": self.best_params
        }, model_path)
        print(f"Best model and metadata saved to: {model_path}")

    def get_best_model(self):
        if self.best_model is None:
            raise RuntimeError(
                "Run find_best_hyperparameters() first to train and "
                "store the model."
            )
        return self.best_model

    def load_model(self):
        """Loads the best model, selected features, and hyperparameters from
        disk."""
        model_path = os.path.join(self.model_dir, "best_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}")

        saved = joblib.load(model_path)
        self.best_model = saved["model"]
        self.selected_features = saved["selected_features"]
        self.best_params = saved["best_params"]
        print(f"Model loaded from {model_path}")