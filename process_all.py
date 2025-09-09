#!/usr/bin/env python3
"""
Simple helper script for batch processing all images in a folder.
This is a simplified version of batch_process.py for common use cases.
"""
import sys
import os
from batch_process import BatchGameAnalysisApp


def main():
    """Simple batch processing with minimal arguments."""
    if len(sys.argv) < 2:
        print("Usage: python process_all.py <image_folder>")
        print("       python process_all.py <image_folder> --save-images")
        print("")
        print("This script will:")
        print("  - Process all images in sorted order")
        print("  - Update gaze data with object and relationship information")
        print("  - Generate visualization images (if --save-images is used)")
        print("")
        print("Example:")
        print("  python process_all.py data/seaquest/54_RZ_2461867_Aug-11-09-35-18")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    save_images = '--save-images' in sys.argv
    
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist")
        sys.exit(1)
    
    print(f"🚀 Starting batch processing of: {image_folder}")
    if save_images:
        print("📷 Will save visualization images")
    
    try:
        app = BatchGameAnalysisApp("seaquest", save_visualizations=save_images)
        app.run_batch(image_folder, fps=1, output_folder=None)
        print("✅ Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
