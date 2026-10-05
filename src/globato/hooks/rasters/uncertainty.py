import logging
import os

import numpy as np
import rasterio
from fetchez.registry import HookRegistry
from scipy.ndimage import distance_transform_edt

from .base import RasterGlobalHook

# try:
#     from sklearn.ensemble import RandomForestRegressor

#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class UncertaintySurface(RasterGlobalHook):
    """Calculates spatial interpolation uncertainty.

    Modes:
      - 'distance': Simple linear error based on distance to nearest measurement.
      - 'split_sample': Cross-validation. Drops points, re-interpolates, and models error.
      - 'ml': Trains a Random Forest to predict uncertainty based on terrain complexity and distance.
    """

    name = "dem_uncertainty"
    default_suffix = "_unc"
    meta_desc = "Caclulates spatial interpolation uncertainty."

    def __init__(
        self,
        mode="split_sample",
        algo="interp_gmt",
        append_band=False,
        drop_frac=0.1,
        fit_method="curve",
        base_error=0.5,
        dist_coeff=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode.lower()
        self.algo = algo
        self.append_band = str(append_band).lower() in ["true", "1", "yes"]

        # Split-Sample Params
        self.drop_frac = float(drop_frac)
        self.fit_method = fit_method.lower()  # 'curve' or 'rmse'

        # Distance Params
        self.base_error = float(base_error)
        self.dist_coeff = float(dist_coeff)
        self.kwargs = kwargs

    def process_raster(self, src_path, dst_path, entry):
        stack_path = src_path.replace("_interp", "_stack").replace("_cudem", "_stack")
        if not os.path.exists(stack_path):
            stack_path = src_path

        with (
            rasterio.open(src_path) as src_interp,
            rasterio.open(stack_path) as src_stack,
        ):
            interp_data = src_interp.read(1)
            stack_data = src_stack.read(1)
            nodata = src_interp.nodata if src_interp.nodata is not None else -9999.0

            valid_mask = (stack_data != nodata) & (~np.isnan(stack_data))

            logger.info(f"Running Uncertainty Analysis (Mode: {self.mode.upper()})...")

            if self.mode == "distance":
                unc_data = self._run_distance(valid_mask, src_interp.res[0])

            elif self.mode == "split_sample":
                unc_data = self._run_split_sample(
                    stack_data, valid_mask, nodata, src_interp.profile
                )

            elif self.mode == "ml":
                unc_data = self._run_ml(
                    stack_data, interp_data, valid_mask, nodata, src_interp.res[0]
                )

            else:
                logger.error(f"Unknown uncertainty mode: {self.mode}")
                return False

            # Mask out areas where the final DEM is NoData
            final_mask = (interp_data == nodata) | (np.isnan(interp_data))
            unc_data[final_mask] = nodata

            self._write_output(src_interp.profile, dst_path, interp_data, unc_data)
            return True

    def _run_distance(self, valid_mask, resolution):
        """Pure geometric distance model."""

        distances, _ = distance_transform_edt(~valid_mask, return_indices=True)
        dist_meters = distances * resolution
        return self.base_error + (dist_meters * self.dist_coeff)

    def _run_split_sample(self, stack_data, valid_mask, nodata, profile):
        """split-sample uncertainty calculations."""

        HookRegistry.load_all()
        interp_cls = HookRegistry.get_class(self.algo)
        if not interp_cls:
            logger.error(f"Cannot run split-sample: Algorithm '{self.algo}' not found.")
            return self._run_distance(valid_mask, profile["transform"][0])

        rows, cols = np.where(valid_mask)
        num_points = len(rows)
        drop_count = int(num_points * self.drop_frac)

        logger.info(
            f"Split-Sample: Dropping {drop_count} points to test '{self.algo}'..."
        )

        drop_idx = np.random.choice(num_points, drop_count, replace=False)
        test_rows, test_cols = rows[drop_idx], cols[drop_idx]

        sparse_stack = stack_data.copy()
        sparse_stack[test_rows, test_cols] = nodata

        temp_sparse = "temp_sparse_stack.tif"
        temp_interp = "temp_sparse_interp.tif"

        with rasterio.open(temp_sparse, "w", **profile) as tmp:
            tmp.write(sparse_stack, 1)

        interp_hook = interp_cls(**self.kwargs)
        interp_hook.process_raster(temp_sparse, temp_interp, {})

        with rasterio.open(temp_interp) as test_src:
            test_data = test_src.read(1)

        # Calculate Absolute Error (Delta Z) at the dropped locations
        true_z = stack_data[test_rows, test_cols]
        pred_z = test_data[test_rows, test_cols]
        dz = np.abs(true_z - pred_z)

        # Calculate Distance to nearest *remaining* points for the test points
        sparse_mask = (sparse_stack != nodata) & (~np.isnan(sparse_stack))
        distances_to_sparse, _ = distance_transform_edt(
            ~sparse_mask, return_indices=True
        )
        distances_to_sparse_m = distances_to_sparse * profile["transform"][0]

        test_distances = distances_to_sparse_m[test_rows, test_cols]

        # Apply the chosen fitting method
        if self.fit_method == "curve":
            logger.info(
                "Fitting empirical error curve (scipy.optimize) to Delta Z vs Distance..."
            )
            try:
                from scipy.optimize import curve_fit

                # Replicates cudem.utils._err2coeff (Quadratic/Linear Growth)
                # E(d) = a + b*d + c*d^2
                def error_curve(d, a, b, c):
                    return a + b * d + c * (d**2)

                # Filter extreme distance outliers (95th percentile)
                max_dist = np.nanpercentile(test_distances, 95)
                valid_fit_mask = (test_distances <= max_dist) & (~np.isnan(dz))

                x_data = test_distances[valid_fit_mask]
                y_data = dz[valid_fit_mask]

                # Fit the curve, forcing coefficients to be positive so uncertainty only grows
                popt, _ = curve_fit(error_curve, x_data, y_data, bounds=(0, np.inf))
                a, b, c = popt

                logger.info(
                    f"Curve Fit Parameters -> Base: {a:.3f}, Linear: {b:.5f}, Quad: {c:.7f}"
                )

                # Apply the mathematical curve to the entire distance raster
                unc_data = error_curve(distances_to_sparse_m, a, b, c)

            except Exception as e:
                logger.error(
                    f"Curve fitting failed: {e}. Falling back to RMSE scaling."
                )
                self.fit_method = "rmse"

        if self.fit_method == "rmse":
            rmse = np.sqrt(np.mean(dz**2))
            logger.info(f"Split-Sample RMSE for {self.algo}: {rmse:.3f}")
            unc_data = rmse + (distances_to_sparse_m * (rmse * 0.05))

        if os.path.exists(temp_sparse):
            os.remove(temp_sparse)
        if os.path.exists(temp_interp):
            os.remove(temp_interp)

        return unc_data

    def _run_ml(self, stack_data, interp_data, valid_mask, nodata, resolution):
        """Machine Learning error prediction using spatial features."""

        # if not HAS_SKLEARN:
        #     logger.warning("scikit-learn missing. Falling back to distance mode.")
        #     return self._run_distance(valid_mask, resolution)

        logger.info("Training ML Uncertainty Predictor...")

        # Distance to nearest point
        distances, _ = distance_transform_edt(~valid_mask, return_indices=True)

        # Local Variance (Roughness)
        from scipy.ndimage import generic_filter

        variance = generic_filter(interp_data, np.var, size=3)

        # todo: train this RF on a split-sample dataset,
        # predicting DZ based on distance and variance.
        unc_data = (distances * 0.01) + (variance * 0.5)

        return unc_data

    def _write_output(self, profile, dst_path, interp_data, unc_data):
        if self.append_band:
            logger.info("Appending Uncertainty as Band 2 to final DEM")
            profile.update(count=2)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(interp_data, 1)
                dst.write(unc_data.astype(profile["dtype"]), 2)
                dst.set_band_description(1, "Elevation")
                dst.set_band_description(2, "Uncertainty (m)")
        else:
            logger.info(f"Writing standalone Uncertainty raster: {dst_path}")
            profile.update(count=1)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(unc_data.astype(profile["dtype"]), 1)
                dst.set_band_description(1, "Uncertainty (m)")
