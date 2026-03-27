"""
COLMAP + GLOMAP Processing Script

Runs Structure-from-Motion (SfM) using COLMAP for feature extraction/matching
and GLOMAP for faster sparse reconstruction.
"""

import os
import subprocess
import argparse
import shutil
from pathlib import Path

# Camera models that have lens distortion parameters.
# graphdeco train.py ignores distortion params during rendering, so training
# on distorted images produces a systematic mismatch. Undistort first.
_DISTORTED_MODELS = {"OPENCV", "OPENCV_FISHEYE", "RADIAL", "FULL_OPENCV",
                     "THIN_PRISM_FISHEYE", "SIMPLE_RADIAL"}


def _run_with_display(cmd):
    """
    Run command with xvfb for headless operation.
    COLMAP requires display even in CPU mode due to Qt/OpenGL initialization.
    """
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "--server-args=-screen 0 1024x768x24"] + cmd
    return cmd


def undistort_images(images_dir, sparse_dir, output_dir, max_image_size=2000):
    """
    Run colmap image_undistorter to convert distorted cameras (e.g. OPENCV)
    to PINHOLE cameras with undistorted images.

    graphdeco train.py expects undistorted images + PINHOLE cameras. This step
    produces an output directory with:
        images/   <- undistorted copies of all input images
        sparse/0/ <- new reconstruction with PINHOLE cameras

    Pass output_dir as -s to graphdeco/train.py.

    Args:
        images_dir: Original input images directory
        sparse_dir: Path to sparse/0 from COLMAP reconstruction
        output_dir: Directory to write undistorted output into
        max_image_size: Cap undistorted image size to avoid huge files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("COLMAP Image Undistortion")
    print("=" * 60)

    cmd = [
        "colmap", "image_undistorter",
        "--image_path",    str(images_dir),
        "--input_path",    str(sparse_dir),
        "--output_path",   str(output_dir),
        "--output_type",   "COLMAP",
        "--max_image_size", str(max_image_size),
    ]
    subprocess.run(_run_with_display(cmd), check=True)

    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    assert images_out.exists(), f"images/ not found in undistorted output: {output_dir}"
    assert sparse_out.exists(), f"sparse/0/ not found in undistorted output: {output_dir}"

    n = len(list(images_out.glob("*")))
    print(f"\nUndistortion complete — {n} images written to {output_dir}")
    return output_dir


def run_colmap(
    images_dir,
    output_dir,
    camera_model="OPENCV",
    quality="high",
    gpu=True,
    use_glomap=True,
    sequential=False,
    sequential_overlap=50,
):
    """
    Run COLMAP + GLOMAP pipeline on images.

    Args:
        images_dir: Directory containing input images
        output_dir: Directory for COLMAP output
        camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
        quality: Processing quality (high, medium, low)
        gpu: Use GPU acceleration
        use_glomap: Use GLOMAP for reconstruction (faster) instead of COLMAP mapper
        sequential: Use sequential matcher instead of exhaustive (faster for ordered video)
        sequential_overlap: Number of neighboring frames to match when sequential=True
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COLMAP Feature Extraction")
    print("=" * 60)
    
    # Feature extraction
    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
    ]
    
    # GPU mode (only add if GPU is enabled and available)
    if gpu:
        feature_cmd.extend(["--SiftExtraction.use_gpu", "1"])
    
    # Set quality parameters
    if quality == "high":
        feature_cmd.extend([
            "--SiftExtraction.max_image_size", "4096",
            "--SiftExtraction.max_num_features", "8192"
        ])
    elif quality == "medium":
        feature_cmd.extend([
            "--SiftExtraction.max_image_size", "2048",
            "--SiftExtraction.max_num_features", "4096"
        ])
    else:  # low
        feature_cmd.extend([
            "--SiftExtraction.max_image_size", "1024",
            "--SiftExtraction.max_num_features", "2048"
        ])
    
    subprocess.run(_run_with_display(feature_cmd), check=True)
    
    print("\n" + "=" * 60)
    if sequential:
        print(f"COLMAP Sequential Matching (overlap={sequential_overlap})")
    else:
        print("COLMAP Exhaustive Matching")
    print("=" * 60)

    if sequential:
        # O(n * overlap) — correct choice for ordered video frames.
        # quadratic_overlap also matches at doubling offsets (i+1, i+2, i+4...)
        # to catch long-range connections when camera revisits a location.
        matching_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", str(sequential_overlap),
            "--SequentialMatching.quadratic_overlap", "1",
        ]
    else:
        # O(n²) — thorough but slow; use for unordered image collections
        matching_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
        ]

    if gpu:
        matching_cmd.extend(["--SiftMatching.use_gpu", "1"])

    subprocess.run(_run_with_display(matching_cmd), check=True)
    
    # Sparse reconstruction - use GLOMAP or COLMAP mapper
    if use_glomap and shutil.which("glomap"):
        print("\n" + "=" * 60)
        print("GLOMAP Sparse Reconstruction (Faster)")
        print("=" * 60)
        
        # GLOMAP mapper (much faster than COLMAP)
        mapper_cmd = [
            "glomap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ]
        
        subprocess.run(_run_with_display(mapper_cmd), check=True)
    else:
        if use_glomap:
            print("\nWarning: GLOMAP not found, falling back to COLMAP mapper")
        
        print("\n" + "=" * 60)
        print("COLMAP Sparse Reconstruction")
        print("=" * 60)
        
        # COLMAP mapper (slower but reliable fallback)
        mapper_cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ]
        
        subprocess.run(_run_with_display(mapper_cmd), check=True)
    
    # Find the reconstruction folder (usually '0')
    reconstruction_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    
    if not reconstruction_dirs:
        raise RuntimeError("COLMAP reconstruction failed - no output generated")
    
    main_reconstruction = reconstruction_dirs[0]
    print(f"\nReconstruction saved to: {main_reconstruction}")
    
    print("\n" + "=" * 60)
    print("COLMAP Processing Complete!")
    print("=" * 60)
    
    return main_reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP Structure-from-Motion on images"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/colmap",
        help="Output directory for COLMAP data"
    )
    parser.add_argument(
        "--no_glomap",
        action="store_true",
        help="Use COLMAP mapper instead of GLOMAP (slower but more compatible)"
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="OPENCV",
        help="Camera model (OPENCV, PINHOLE, RADIAL, etc.)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["high", "medium", "low"],
        help="Processing quality (high, medium, low)"
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential matcher instead of exhaustive (faster for ordered video frames)"
    )
    parser.add_argument(
        "--sequential_overlap",
        type=int,
        default=50,
        help="Number of neighboring frames to match when using --sequential (default: 50)"
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Run colmap image_undistorter after reconstruction (required when camera model "
             "has distortion params and training with graphdeco)"
    )
    parser.add_argument(
        "--undistort_output",
        type=str,
        default=None,
        help="Output directory for undistorted images + PINHOLE cameras "
             "(default: <output_dir>_undistorted)"
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.images_dir).exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    # Run COLMAP + GLOMAP
    recon_dir = run_colmap(
        args.images_dir,
        args.output_dir,
        camera_model=args.camera_model,
        quality=args.quality,
        gpu=not args.no_gpu,
        use_glomap=not args.no_glomap,
        sequential=args.sequential,
        sequential_overlap=args.sequential_overlap,
    )

    # Optionally undistort
    if args.undistort:
        if args.camera_model not in _DISTORTED_MODELS:
            print(f"Note: --undistort passed but {args.camera_model} has no distortion "
                  f"params — skipping.")
        else:
            undistort_output = args.undistort_output or str(Path(args.output_dir).parent /
                                                             (Path(args.output_dir).name + "_undistorted"))
            undistort_images(args.images_dir, recon_dir, undistort_output)


if __name__ == "__main__":
    main()
