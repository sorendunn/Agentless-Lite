import os
import subprocess

import git


def get_repo_folder_name(repo_url):
    """Infer the top folder name from the repository URL"""
    return repo_url.split("/")[-1]


def ensure_repo_exists(repo, testbed_dir, logger):
    """Clone the repository if it doesn't exist"""
    repo_path = os.path.join(testbed_dir, get_repo_folder_name(repo))
    if not os.path.exists(repo_path):
        logger.info(f"Cloning repository {repo} to {repo_path}")
        git.Repo.clone_from(f"https://github.com/{repo}.git", repo_path)
    return repo_path


def remove_outer_folder(path):
    _, remaining_path = os.path.split(os.path.dirname(path))
    result = os.path.join(remaining_path, os.path.basename(path))
    return result


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    First discards any untracked changes in the repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        print(f"Cleaning untracked files in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "clean", "-fd"], check=True)

        print("Discarding changes in tracked files...")
        subprocess.run(["git", "-C", repo_path, "reset", "--hard"], check=True)

        print(f"Checking out commit {commit_id}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_all_file_paths(folder_path):
    """
    Returns a list of absolute file paths for all files in the given folder and its subfolders.

    Args:
        folder_path (str): The path to the folder to search

    Returns:
        list: A list of absolute file paths
    """
    file_paths = []

    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            repo_file_path = os.path.relpath(file_path, folder_path)
            file_paths.append(repo_file_path)

    return file_paths
