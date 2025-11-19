"""
rename.py

Helper script to batch-rename files in a directory, replacing 'top' with 'bot' in filenames.
Intended for dataset organization and correction of labeling errors.
"""

import os

DIRECTORY = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\Classifier\validation\bot_mix"


def main(directory: str = DIRECTORY) -> None:
    """
    Batch-renames files in the given directory, replacing 'top' with 'bot' in filenames.

    Args:
        directory (str): Path to the target directory.
    """
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        if os.path.isfile(src):
            new_filename = filename.replace("top", "bot")
            dst = os.path.join(directory, new_filename)
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    import sys

    dir_arg = sys.argv[1] if len(sys.argv) > 1 else DIRECTORY
    main(dir_arg)
