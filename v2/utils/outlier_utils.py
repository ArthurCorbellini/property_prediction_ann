from feature_engine.outliers import OutlierTrimmer


class OutlierUtils:
    """
    Util class used to get the default outlier treatment.
    """

    @staticmethod
    def trimmer_iqr(variables):
        """
        Returns a default OulierTrimmer for skewed distribution.
        """
        return OutlierTrimmer(
            variables=variables,
            tail="both",
            capping_method="iqr",
            fold=1.5,
        )

    @staticmethod
    def trimmer_normal_gaussian(variables):
        """
        Returns a default OulierTrimmer for normal distribution.
        - Gaussian method based.
        """
        return OutlierTrimmer(
            variables=variables,
            tail="both",
            capping_method="gaussian",
            fold=3,
        )

    @staticmethod
    def trimmer_normal_quantile(variables):
        """
        Returns a default OulierTrimmer for normal distribution.
        - Quantile range based;
        - Its more agressive than Gaussian.
        """
        return OutlierTrimmer(
            variables=variables,
            capping_method="quantiles",
            tail="both",
            fold=0.05,
        )
