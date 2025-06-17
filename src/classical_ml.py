import os
import numpy as np
import pandas as pd
import cv2 as cv
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils import get_paths, read_image


MODELS = {
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM": SVC(),
    "NB": GaussianNB(),
    "RF": RandomForestClassifier(),
}


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
            img_8bit = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        else:
            img_8bit = img

        ret, _ = cv.threshold(img_8bit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        actual_otsu_threshold = ret

        pixels_above_thr = img[img >= actual_otsu_threshold]
        pixels_below_thr = img[img < actual_otsu_threshold]

        variance_sum = 0.0
        if pixels_above_thr.size > 0:
            if pixels_above_thr.size > 1:
                variance_sum += np.mean(pixels_above_thr) * np.var(pixels_above_thr)
            else:
                variance_sum += np.mean(pixels_above_thr) * 0.0

        if pixels_below_thr.size > 0:
            if pixels_below_thr.size > 1:
                variance_sum += np.mean(pixels_below_thr) * np.var(pixels_below_thr)
            else:
                variance_sum += np.mean(pixels_below_thr) * 0.0

        return actual_otsu_threshold

    def _count_edges(self, img):
        """Counts the number of edges in the image using Canny edge detection."""
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

# class ClassicalML:
#     def __init__(
#             self,
#             model_type: str,
#             normal_path: str,
#             pneumonia_path: str,
#             models_path: str,
#     ):
#         self.model_type = model_type
#         self.model = MODELS[self.model_type]
#         self.normal_path = normal_path
#         self.pneumonia_path = pneumonia_path
#         self.models_path = models_path