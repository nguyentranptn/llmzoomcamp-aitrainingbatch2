import os
import re


def sanitize_filename(path: str) -> str:
    """
    Make file path safe for Windows by removing invalid characters.
    """
    directory, filename = os.path.split(path)

    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    return os.path.join(directory, filename)
