import os
import platform
import subprocess


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
            return False, f"Unsupported file type: {file_extension}"

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
    """Check JavaScript syntax using node"""
    try:
        if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
            return False, "Node.js is not installed"

        command = f'node --check "{filepath}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return True, "Syntax is valid"
        else:
            return False, f"Syntax error: {result.stderr}"
    except Exception as e:
        return False, f"Error checking JavaScript syntax: {str(e)}"


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
    ]

    for file in test_files:
        print(f"\nTesting {file}:")
        is_valid, message = check_syntax(file)
        print(f"Valid: {is_valid}")
        print(f"Message: {message}")


if __name__ == "__main__":
    test_syntax_checker()
