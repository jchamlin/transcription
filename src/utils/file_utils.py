def read_file_lines(filename, encoding="utf-8"):
    """
    Reads all lines from a text file into a list, stripping newline characters.

    Args:
        filename (str): Path to the file to read.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        list[str]: Lines from the file, with trailing newlines removed.
    """
    with open(filename, "r", encoding=encoding) as f:
        return [line.rstrip("\n") for line in f]

def write_file_lines(filename, lines, encoding="utf-8", newline="\n"):
    """
    Writes a list of strings to a text file, adding a newline after each line.

    Args:
        filename (str): Path to the file to write.
        lines (list[str]): List of strings to write to the file.
        encoding (str, optional): Encoding to use when writing the file. Defaults to "utf-8".
        newline (str, optional): Newline character(s) to use. Defaults to "\n".
    """
    with open(filename, "w", encoding=encoding) as f:
        f.writelines(line + newline for line in lines)

def read_file(filename, encoding="utf-8"):
    """
    Reads the entire content of a text file as a single string.

    Args:
        filename (str): Path to the file to read.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        str: The full content of the file as a single string.
    """
    with open(filename, "r", encoding=encoding) as f:
        return f.read()


def write_file(filename, content, encoding="utf-8", newline="\n"):
    """Writes a string to a text file using the specified encoding and newline format.

    Args:
        filename (str): Path to the file to write.
        content (str): Full text content to write into the file.
        encoding (str, optional): Encoding to use when writing the file. Defaults to "utf-8".
        newline (str, optional): Newline character(s) to use. Defaults to "\n".
    """
    with open(filename, "w", encoding=encoding, newline=newline) as f:
        f.write(content)
