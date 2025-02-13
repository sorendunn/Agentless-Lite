import os
import re
import subprocess
import uuid
from difflib import get_close_matches
from typing import Optional

import tiktoken

from agentless_lite.util.syntax import check_syntax
from agentless_lite.util.utils import find_consecutive_subset


def num_tokens_from_messages(message, model="gpt-4o"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def extract_code_blocks(text):
    pattern = r"```(\w+)\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match[1] for match in matches]


def parse_edit_command(edit_string):
    """
    Parse an edit command string to extract filename and code blocks.

    Args:
        edit_string (str): The edit command string in the specified format

    Returns:
        tuple: (filename, search_code, replace_code)
    """
    try:
        lines = edit_string.rstrip().split("\n")

        if not lines:
            print("Error: Empty input")
            return "", "", ""

        filename = lines[0].replace("#", "").strip()

        search_start = -1
        replace_marker = -1
        replace_end = -1

        for i, line in enumerate(lines):
            if line.rstrip().endswith("SEARCH") and line.strip().startswith("<"):
                search_start = i
            # Look for "=======" with any number of '=' characters
            elif (
                line.strip() and all(c == "=" for c in line.strip()) and ("==" in line)
            ):
                replace_marker = i
            # Look for ">+ REPLACE" with any number of '>' characters
            elif line.rstrip().endswith("REPLACE") and line.strip().startswith(">"):
                replace_end = i

        print(search_start, replace_marker, replace_end)

        if search_start == -1 or replace_marker == -1 or replace_end == -1:
            print("Error: Missing markers")
            return "", "", ""

        if not (search_start < replace_marker < replace_end):
            print("Error: Markers are in incorrect order")
            return "", "", ""

        search_code = "\n".join(lines[search_start + 1 : replace_marker]).rstrip()
        replace_code = "\n".join(lines[replace_marker + 1 : replace_end]).rstrip()

        return filename, search_code, replace_code

    except Exception as e:
        print(f"Error parsing edit command: {str(e)}")
        return "", "", ""


def apply_edit_commands(
    parsed_edit_commands, contents, files, match_fuzzy, test_each=False
):
    assert len(files) == len(contents), (
        f"Input lists to apply_edit_commands must have same length. "
        f"They have lengths: {len(files)} and {len(contents)}"
    )

    new_contents = []
    original_file_contents = ""
    for idx, file_name in enumerate(files):
        new_content = contents[idx]
        for file, original, replacement in parsed_edit_commands:
            if file_name == file:
                original_file_contents = contents[idx]
                # First try exact match
                if "\n" + original in new_content:
                    if test_each:
                        temp_new_content = new_content.replace(original, replacement)
                        print("checking an individual")

                        git_diff = fake_git_repo(
                            "playground", ["test_file.py"], [""], [temp_new_content]
                        )
                        if git_diff != "":
                            print("individual worked!")
                            new_content = temp_new_content

                    else:
                        new_content = new_content.replace(original, replacement)
                        print("Found exact match")
                elif match_fuzzy:
                    chunked = new_content.splitlines()
                    chunked_edit = original.splitlines()

                    if chunked_edit[0].strip() == "":
                        chunked_edit = chunked_edit[1:]
                    if chunked_edit[-1].strip == "":
                        chunked_edit = chunked_edit[:-1]

                    matching_line_numbers = []
                    for line in chunked_edit:
                        print(line)
                        matches = get_close_matches(
                            line.strip(),
                            [chunk.strip() for chunk in chunked],
                            n=1,
                            cutoff=0.8,
                        )
                        print(matches)
                        if matches:
                            line_numbers = [
                                i
                                for i, text in enumerate(chunked)
                                if text.strip() in matches
                            ]
                            matching_line_numbers.extend(line_numbers)

                    empty_lines = []
                    for i, line_text in enumerate(chunked):
                        if line_text.strip() == "":
                            empty_lines.append(i)

                    # Add in all the empty lines too
                    for each_line in chunked_edit:
                        if each_line.strip() == "":
                            matching_line_numbers.extend(empty_lines)
                            break

                    matched_line_numbers = find_consecutive_subset(
                        matching_line_numbers, len(chunked_edit), empty_lines
                    )
                    if matched_line_numbers:
                        replaced_indent = len(chunked[matched_line_numbers[0]]) - len(
                            chunked[matched_line_numbers[0]].lstrip()
                        )
                        replacement_indent = len(chunked_edit[0]) - len(
                            chunked_edit[0].lstrip()
                        )

                        new_replacement_text = replacement.splitlines()

                        fixed_replacement = []

                        if replacement_indent < replaced_indent:
                            for new_line in new_replacement_text:
                                fixed_replacement.append(
                                    " " * (replaced_indent - replacement_indent)
                                    + new_line
                                )
                        elif replacement_indent > replaced_indent:
                            for new_line in new_replacement_text:
                                fixed_replacement.append(
                                    new_line[(replacement_indent - replaced_indent) :]
                                )
                        else:
                            fixed_replacement = new_replacement_text

                        fixed_replacement_str = "\n".join(fixed_replacement)
                        fixed_search_str = "\n".join(
                            [
                                chunked_line
                                for idx, chunked_line in enumerate(
                                    new_content.splitlines()
                                )
                                if idx in matched_line_numbers
                            ]
                        )

                        new_content = new_content.replace(
                            fixed_search_str, fixed_replacement_str
                        )

        new_contents.append(new_content)

    return new_contents, original_file_contents


def fake_git_repo(repo_playground, file_paths, old_contents, new_contents) -> str:
    """create a fake git repo to obtain git diff format for multiple files"""
    assert (
        len(file_paths) == len(old_contents) == len(new_contents)
    ), f"Input lists must have same length. They have lengths: {len(file_paths)}, {len(old_contents)}, and {len(new_contents)}"

    repo_playground = os.path.join(repo_playground, f"{uuid.uuid4()}")

    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    os.makedirs(repo_playground)
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    changed_files = []
    for file_path, old_content, new_content in zip(
        file_paths, old_contents, new_contents
    ):
        if old_content != new_content:
            # create directory if needed
            subprocess.run(
                f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
            )
            # write old content
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(old_content)
            changed_files.append(file_path)

    if not changed_files:
        print("No changes were made")
        # No changes to commit, clean up and return empty string
        subprocess.run(f"rm -rf {repo_playground}", shell=True)
        return ""

    # add files to git
    changed_files_str = " ".join(changed_files)
    subprocess.run(
        f"cd {repo_playground} && git add {changed_files_str} && git commit -m 'initial commit'",
        shell=True,
    )

    # edit files with new content
    for file_path, old_content, new_content in zip(
        file_paths, old_contents, new_contents
    ):
        if old_content != new_content:
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(new_content)
            if not check_syntax(f"{repo_playground}/{file_path}")[0]:
                print("failed syntax check")
                with open(f"{repo_playground}/{file_path}", "w") as f:
                    f.write(old_content)

    # get git diff for changed files
    o = subprocess.run(
        f"cd {repo_playground} && git diff {changed_files_str}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s


def create_diff_from_response(
    response: str, old_contents: list, files: list
) -> Optional[str]:
    extracted_python_blocks = extract_code_blocks(response)
    try:
        edits = [
            parse_edit_command(edit_command) for edit_command in extracted_python_blocks
        ]
        new_contents, _ = apply_edit_commands(edits, old_contents, files, False, False)
        git_diff = fake_git_repo("playground", files, old_contents, new_contents)
        return git_diff if git_diff.strip() else None
    except:
        return None
