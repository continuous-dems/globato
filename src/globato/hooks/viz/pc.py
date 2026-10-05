#!/usr/bin/env python

"""
globato.hooks.viz.pc
~~~~~~~~~~~~~~~~~~~~~~~

point cloud visualizations
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn

# from mpl_toolkits.mplot3d import Axes3D
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class PointCloudViz(FetchHook):
    name = "viz_point_cloud"
    meta_stage = "file"
    meta_category = "stream-sink"
    meta_desc = "Visualize a point cloud."

    def __init__(
        self,
        output="{base}_viz.png",
        outliers=False,
        is_3d=False,
        artifact_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output = output
        self.outliers = outliers
        self.is_3d = is_3d
        self.artifact_id = artifact_id or self.name

    def run(self, entries):
        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue

            stream = entry.get("stream")
            region = getattr(mod, "region", None)
            if not region:
                continue

            src_fn = entry.get("dst_fn", "unknown")
            base = os.path.splitext(os.path.basename(src_fn))[0]
            out_fn = self.output.format(base=base, name=mod.name)

            if not os.path.isabs(out_fn):
                out_dir = (
                    os.path.dirname(src_fn) if src_fn != "unknown" else os.getcwd()
                )
                out_fn = os.path.join(out_dir, out_fn)

            entry["stream"] = self._viz_stream(
                stream, out_fn, outliers=self.outliers, is_3d=self.is_3d
            )
            # entry.setdefault('artifacts', {})[self.name] = out_fn
            entry.setdefault("artifacts", {})[self.artifact_id] = out_fn

        return entries

    def _viz_stream(
        self,
        stream,
        out_fn,
        max_points=100000,
        cmap="viridis",
        outliers=False,
        is_3d=False,
    ):
        logger.info(f"Loading point cloud from stream: {stream}...")

        processed_chunks = []
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                chunk = rfn.append_fields(
                    chunk,
                    "classification",
                    np.zeros(len(chunk), dtype=int),
                    usemask=False,
                )
            processed_chunks.append(chunk)

            yield chunk

        if not processed_chunks:
            logger.error("No points found.", fg="red")

        points = rfn.stack_arrays(processed_chunks, asrecarray=True, usemask=False)
        total_pts = len(points)
        logger.info(f"Ready to render {total_pts:,} points.")

        fig = plt.figure(figsize=(10, 8))

        if outliers:
            logger.info("Rendering Outlier Showcase...")
            ax = fig.add_subplot(111)
            noise_mask = points["classification"] == 7
            valid_pts, noise_pts = points[~noise_mask], points[noise_mask]

            if len(valid_pts) > 0:
                ax.scatter(
                    valid_pts["x"],
                    valid_pts["y"],
                    c="lightgray",
                    s=2,
                    alpha=0.5,
                    label=f"Valid ({len(valid_pts)} points)",
                )
            if len(noise_pts) > 0:
                ax.scatter(
                    noise_pts["x"],
                    noise_pts["y"],
                    c="red",
                    s=0.5,
                    marker="x",
                    label=f"Rejected ({len(noise_pts)} points)",
                )

            ax.set_title(f"Filter Results: {len(noise_pts):,} Outliers Found")
            ax.legend()
            ax.set_aspect("equal", "datalim")

        elif is_3d:
            logger.info("Rendering Interactive 3D View...")
            ax = fig.add_subplot(111, projection="3d")
            render_pts = points
            if total_pts > max_points:
                logger.info(
                    f"Decimating to {max_points:,} points for 3D performance..."
                )
                indices = np.random.choice(total_pts, max_points, replace=False)
                render_pts = points[indices]

            p = ax.scatter(
                render_pts["x"],
                render_pts["y"],
                render_pts["z"],
                c=render_pts["z"],
                cmap=cmap,
                s=2,
                alpha=0.8,
            )
            fig.colorbar(p, ax=ax, label="Elevation (Z)")
            ax.set_title(f"3D Sanity Check ({len(render_pts):,} points)")

        else:
            logger.info(" Rendering 2D Top-Down View...")
            ax = fig.add_subplot(111)
            if total_pts > max_points:
                hb = ax.hexbin(
                    points["x"],
                    points["y"],
                    C=points["z"],
                    gridsize=100,
                    cmap=cmap,
                    reduce_C_function=np.mean,
                )
                fig.colorbar(hb, ax=ax, label="Mean Elevation (Z)")
            else:
                sc = ax.scatter(points["x"], points["y"], c=points["z"], cmap=cmap, s=2)
                fig.colorbar(sc, ax=ax, label="Elevation (Z)")
            ax.set_title(f"2D Elevation Map ({total_pts:,} points)")
            ax.set_aspect("equal", "datalim")

        plt.tight_layout()
        plt.savefig(out_fn)
        plt.show()
