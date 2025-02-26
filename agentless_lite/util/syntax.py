import os
import platform
import subprocess

import tree_sitter_javascript
from tree_sitter import Language, Parser


def check_syntax(filepath):
    """
    Check the syntax of a code file.

    Args:
        filepath (str): Path to the code file

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    file_extension = os.path.splitext(filepath)[1].lower()

    try:
        if file_extension in [".py", ".pyw"]:
            return check_python_syntax(filepath)

        elif file_extension in [".js", ".jsx"]:
            return check_javascript_syntax(filepath)

        elif file_extension in [".sh", ".bash"]:
            return check_shell_syntax(filepath)

        else:
            return True, f"Unsupported file type: {file_extension}"

    except Exception as e:
        return False, f"Error during syntax check: {str(e)}"


def check_python_syntax(filepath, timeout_seconds=30):
    """Check Python syntax using py_compile with timeout"""
    try:
        command = f'python -m py_compile "{filepath}"'
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_seconds
        )

        if result.returncode == 0:
            return True, "Syntax is valid"
        else:
            return False, f"Syntax error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"Process timed out after {timeout_seconds} seconds"
    except Exception as e:
        return False, f"Error checking Python syntax: {str(e)}"


def check_javascript_syntax(filepath):
    """Check JavaScript/JSX syntax using tree-sitter"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            # Initialize tree-sitter parser with JavaScript language
            JS_LANGUAGE = Language(tree_sitter_javascript.language())
            parser = Parser(JS_LANGUAGE)

            # Parse the code
            tree = parser.parse(bytes(code, "utf8"))

            # If there are syntax errors, tree-sitter will include ERROR nodes
            has_errors = any(node.type == "ERROR" for node in tree.root_node.children)

            if has_errors:
                return False, "Syntax error detected in the code"
            return True, "Syntax is valid"

        except Exception as e:
            return False, f"Syntax error: {str(e)}"

    except ImportError:
        return (
            False,
            "tree-sitter-javascript is not installed. Install it using: pip install tree-sitter-javascript",
        )
    except Exception as e:
        return False, f"Error checking JavaScript/JSX syntax: {str(e)}"


def check_shell_syntax(filepath):
    """Check shell script syntax using bash -n"""
    try:
        if platform.system() == "Windows":
            shell_cmd = "bash"
        else:
            shell_cmd = "bash"

        command = f'{shell_cmd} -n "{filepath}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return True, "Syntax is valid"
        else:
            return False, f"Syntax error: {result.stderr}"
    except Exception as e:
        return False, f"Error checking shell script syntax: {str(e)}"


def test_syntax_checker():
    test_files = [
        "valid.py",
        "invalid.py",
        "valid.js",
        "invalid.js",
        "valid.sh",
        "invalid.sh",
        "output_files/Automattic__wp-calypso-22242/main.jsx",
    ]

    for file in test_files:
        print(f"\nTesting {file}:")
        is_valid, message = check_syntax(file)
        print(f"Valid: {is_valid}")
        print(f"Message: {message}")


if __name__ == "__main__":
    test_syntax_checker()
