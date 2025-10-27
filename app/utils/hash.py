import hashlib


def compute_md5_file(file_path):
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_md5_string(input_string):
    """Compute the MD5 hash of a string."""
    return hashlib.md5(input_string.encode("utf-8")).hexdigest()
