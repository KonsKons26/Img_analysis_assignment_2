import os
import numpy as np
import cv2 as cv
from skimage.feature import hog


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

    FEATURE_NAMES = [
        "mean", "median", "std", "entropy", "energy",
        
        "hist_means_0_50", "hist_means_51_100", "hist_means_101, 150",
        "hist_means_151_200", "hist_means_201_250",
        "hist_std_0_50", "hist_std_51_100", "hist_std_101_150",
        "hist_std_151_200", "hist_std_201_250",

        "otsu_threshold",

        "n_edges",
        
        "horizontal_lines", "vertical_lines",

        "hog_features",

        "fft_mean", "fft_std",

        "fractal_dim",

        "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
        "hu_moment_5", "hu_moment_6", "hu_moment_7"
    ]

    def __init__(
            self,
            img,
            otsu_thr,
            edge_thr1, edge_thr2,
            line_thr, hor_thr, ver_thr,
            hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
            frac_thr
    ):
        self.img = img
        self.otsu_thr = otsu_thr
        self.edge_thr1 = edge_thr1
        self.edge_thr2 = edge_thr2
        self.line_thr = line_thr
        self.hor_thr = hor_thr
        self.ver_thr = ver_thr
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.frac_thr = frac_thr

    def extract_features(self):
        # General stats
        mean, median, std, entropy, energy = self._extract_first_order_stats()

        # Histogram stats
        hist_means, hist_stds = self._extract_hist_stats()

        # Otsu's threshold
        otsus_threshold = self._otsu_intraclass_variance()

        # Number of edges
        edges_count = self._count_edges()

        # Horizontal and vertical lines
        hor, ver = self._count_horizontals_verticals()

        # HOG features
        hog_features = self._extract_hog_features()

        # FFT mean and std
        fft_mean, fft_std = self._extract_fft_features()

        # Fractal dim
        fractal_dim = self._fractal_dimension()

        # Hu's seven invariant moments
        hu_moments = self._extract_hu_moments()

        return np.array([
            mean, median, std, entropy, energy,

            hist_means[0], hist_means[1], hist_means[2], hist_means[3],
            hist_means[4], hist_stds[0], hist_stds[1], hist_stds[2],
            hist_stds[3], hist_stds[4],

            otsus_threshold,

            edges_count,

            hor, ver,

            hog_features,

            fft_mean, fft_std,

            fractal_dim,

            hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
            hu_moments[4], hu_moments[5], hu_moments[6]
        ])

    def _extract_first_order_stats(self):
        return [
            np.mean(self.img),
            np.median(self.img),
            np.std(self.img),
            -np.sum(self.img * np.log2(self.img + 1e-10)),
            np.sum(self.img**2)
        ]

    def _extract_hist_stats(self):
            hist = cv.calcHist([self.img], [0], None, [256], [0, 256])
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

    def _otsu_intraclass_variance(self):
        return np.nansum(
            [
                np.mean(cls) * np.var(self.img, where=cls)
                for cls in [self.img >= self.otsu_thr, self.img < self.otsu_thr]
            ]
        )

    def _count_edges(self):
        img = self.img.copy()
        img = cv.GaussianBlur(img, (7, 7), 0)
        edges = cv.Canny(img, self.edge_thr1, self.edge_thr2)
        self.edges = edges
        return np.sum(edges > 0)

    def _count_horizontals_verticals(self):
        hor, ver = 0, 0
        im = self.edges.copy()
        lines = cv.HoughLines(im, 1, np.pi/180, self.line_thr)
        if lines is not None:
            for line in lines:
                for _, theta in line:
                    angle = theta * 180 / np.pi

                    if (
                        90 - self.hor_thr < angle < 90 + self.hor_thr
                    ) or (
                        270 - self.hor_thr < angle < 270 + self.hor_thr
                    ):
                        hor += 1

                    if (
                        0 - self.ver_thr < angle < 0 + self.ver_thr
                    ) or (
                        180 - self.ver_thr < angle < 180 + self.ver_thr
                    ):
                        ver += 1

        return hor, ver

    def _extract_hog_features(self):
        features = hog(
            self.img,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True
        )
        return features.shape[0]

    def _extract_fft_features(self):
        fft = np.fft.fft2(self.img)
        magnitude = np.abs(fft)
        return [
            np.mean(magnitude),
            np.std(magnitude)
        ]

    def _fractal_dimension(self):
        def boxcount(image, k):
            S = np.add.reduceat(
                np.add.reduceat(image, 
                            np.arange(0, image.shape[0], k), 
                            axis=0),
                np.arange(0, image.shape[1], k), 
                axis=1)
            return len(np.where(S > 0)[0])
        
        binary = (self.img > self.frac_thr).astype(int)
        sizes = 2**np.arange(1, 8)
        counts = [boxcount(binary, size) for size in sizes]
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def _extract_hu_moments(self):
        moments = cv.moments(self.img)
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