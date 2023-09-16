from feature_engine.outliers import OutlierTrimmer


def trimmer_skewed_iqr(variables, x_train, x_test, y_train, y_test):
    """
    Apply a default OulierTrimmer for skewed distribution.
    """
    return _build_trimmer(variables, 'iqr', 1.5, x_train, x_test, y_train, y_test)


def trimmer_normal_gaussian(variables, x_train, x_test, y_train, y_test):
    """
    Apply a default OulierTrimmer for normal distribution.
    - Gaussian method based.
    """
    return _build_trimmer(variables, 'gaussian', 3, x_train, x_test, y_train, y_test)


def trimmer_normal_quantile(variables, x_train, x_test, y_train, y_test):
    """
    Apply a default OulierTrimmer for normal distribution.
    - Quantile range based;
    - Its more agressive than Gaussian.
    """
    return _build_trimmer(variables, 'quantiles', 0.01, x_train, x_test, y_train, y_test)


def _build_trimmer(variables, capping_method, fold, x_train, x_test, y_train, y_test):
    trimmer = OutlierTrimmer(
        variables=variables,
        capping_method=capping_method,
        tail='both',
        fold=fold
    )
    trimmer.fit(x_train)
    # print(trimmer.left_tail_caps_)
    # print(trimmer.right_tail_caps_)
    x_train_trimmed = trimmer.transform(x_train)
    y_train_trimmed = y_train[x_train_trimmed.index]
    x_test_trimmed = trimmer.transform(x_test)
    y_test_trimmed = y_test[x_test_trimmed.index]
    return x_train_trimmed, x_test_trimmed, y_train_trimmed, y_test_trimmed


def trimmer_skewed_iqr(variables, data_frame):
    return _build_trimmer(variables, 'iqr', 1.5, data_frame)


def trimmer_normal_gaussian(variables, data_frame):
    return _build_trimmer(variables, 'gaussian', 3, data_frame)


def trimmer_normal_quantile(variables, data_frame):
    return _build_trimmer(variables, 'quantiles', 0.01, data_frame)


def _build_trimmer(variables, capping_method, fold, data_frame):
    return OutlierTrimmer(
        variables=variables,
        capping_method=capping_method,
        tail='both',
        fold=fold
    ).fit_transform(data_frame)
