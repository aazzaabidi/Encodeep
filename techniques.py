


class GADF(BaseEstimator, TransformerMixin):

    def __init__(self, image_size=32, overlapping=False, scale=-1):
        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        # Check input data
        X = check_array(X)
        # Shape parameters
        n_samples, n_features = X.shape
        # Check parameters
        paa = PAA(output_size=self.image_size, overlapping=self.overlapping)
        X_paa = paa.fit_transform(X)
        n_features_new = X_paa.shape[1]
        scaler = MinMaxScaler(feature_range=(self.scale, 1))
        X_scaled = scaler.fit_transform(X_paa.T).T
        X_sin = np.sqrt(np.clip(1 - X_scaled**2, 0, 1))
        X_scaled_sin = np.hstack([X_scaled, X_sin])
        X_scaled_sin_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 True)
        X_sin_scaled_outer = np.apply_along_axis(self._outer_stacked,
                                                 1,
                                                 X_scaled_sin,
                                                 n_features_new,
                                                 False)
        return X_sin_scaled_outer - X_scaled_sin_outer
    def _outer_stacked(self, arr, size, first=True):
        if first:
            return np.outer(arr[:size], arr[size:])
        else:
            return np.outer(arr[size:], arr[:size])

####################################################################################################################################
