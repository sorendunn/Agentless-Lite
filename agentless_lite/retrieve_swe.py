import argparse
import concurrent
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

import pandas as pd
import weave

from agentless_lite.util.logging import setup_logging
from agentless_lite.util.repo import (
    checkout_commit,
    ensure_repo_exists,
    get_all_file_paths,
)
from agentless_lite.util.retrieve import get_embedding_model, retrieve

RETRIEVAL_INSTRUCTION = (
    "\n\nFind code the code which need to be edited to solve the above issue"
)


def retrieve_instance(args, row, output_path, repo_locks, output_lock):
    """Process a single row from the DataFrame"""

    log_dir = os.path.join(args.output_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(row["instance_id"], log_dir)
    logger.info(f"Starting retrieval for instance {row['instance_id']}")

    try:
        retrieval_model = get_embedding_model(args.embedding_model, logger)

        original_prompt = row["problem_statement"]
        persist_dir = os.path.join(args.embedding_folder, row["instance_id"])

        logger.info(f"Using persist directory: {persist_dir}")

        with repo_locks[row["repo"]]:
            file_dir = ensure_repo_exists(row["repo"], args.testbed_dir, logger)
            logger.info(
                f"Checking out commit {row['base_commit']} for repo {row['repo']}"
            )
            checkout_commit(file_dir, row["base_commit"])
            files = get_all_file_paths(file_dir)

            if args.filter_python:
                files = [
                    file_path
                    for file_path in files
                    if file_path.endswith(".py") and (not "/test" in file_path)
                ]

            logger.info(f"Found {len(files)} relevant files")

            file_to_contents = {}
            for file in files:
                try:
                    with open(os.path.join(file_dir, file), "r") as f:
                        file_to_contents[file] = f.read()
                except Exception as e:
                    logger.error(
                        f"Failed to load file contents for file {file}: {str(e)}"
                    )

        logger.info("Starting retrieval process")
        file_names, file_contents = retrieve(
            file_to_contents.keys() if file_to_contents else [],
            original_prompt + RETRIEVAL_INSTRUCTION,
            retrieval_model,
            persist_dir,
            args.filter_num,
            args.retrieve_num,
            file_to_contents,
            args.entire_file,
            args.just_create_index,
            logger=logger,
            filter_model_name=args.filter_model,
        )

        logger.info(f"Retrieved {len(file_names)} files")

        result = {
            "instance_id": row["instance_id"],
            "problem_description": original_prompt,
            "found_files": file_names,
            "file_contents": file_contents,
        }

        with output_lock:
            with open(output_path, "a") as f:
                json_line = json.dumps(result)
                f.write(json_line + "\n")
                f.flush()
            logger.info("Successfully wrote results to output file")

        logger.info("Completed retrieval and saved results")
        return result

    except Exception as e:
        logger.error(f"Error processing instance {row['instance_id']}: {str(e)}")
        raise


def retrieve_swe(args):
    try:
        os.makedirs(args.output_folder, exist_ok=True)

        splits = {
            "dev": "data/dev-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        }

        print(f"Loading dataset from {args.dataset}")
        swe_df = pd.read_parquet(f"hf://datasets/{args.dataset}/" + splits["test"])
        print(f"Loaded {len(swe_df)} instances")

        repo_locks = {repo: threading.Lock() for repo in swe_df["repo"].unique()}
        output_lock = threading.Lock()

        output_path = os.path.join(args.output_folder, args.output_file)
        with open(output_path, "w") as f:
            pass  # Just create/clear the file
        print(f"Created output file: {output_path}")

        if args.num_threads == 1:
            print("Running in single-threaded mode")
            for _, row in swe_df.iterrows():
                retrieve_instance(args, row, output_path, repo_locks, output_lock)
        else:
            print(f"Running with {args.num_threads} threads")
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                repo_groups = {}
                for _, row in swe_df.iterrows():
                    if row["repo"] not in repo_groups:
                        repo_groups[row["repo"]] = []
                    repo_groups[row["repo"]].append(row)

                for repo in repo_groups:
                    shuffle(repo_groups[repo])

                rows_list = []
                while any(len(group) > 0 for group in repo_groups.values()):
                    for repo in list(repo_groups.keys()):
                        if repo_groups[repo]:
                            rows_list.append(repo_groups[repo].pop())

                future_to_row = {
                    executor.submit(
                        retrieve_instance,
                        args,
                        row,
                        output_path,
                        repo_locks,
                        output_lock,
                    ): row
                    for row in rows_list
                }

                for future in concurrent.futures.as_completed(future_to_row):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Thread execution failed: {str(e)}")

        print("Completed all retrievals")

    except Exception as e:
        print(f"Fatal error during processing: {str(e)}")
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process SWE-bench dataset with concurrent retrieval."
    )
    parser.add_argument(
        "--embedding_folder",
        type=str,
        default="embeddings",
        help="Base path for the project",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Maximum number of concurrent workers",
    )
    parser.add_argument(
        "--filter_num",
        type=int,
        default=300,
        help="Number of snippets to initially filter down to using the filter model",
    )
    parser.add_argument(
        "--retrieve_num",
        type=int,
        default=100,
        help="Number of snippets to retrieve after filtering",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Output folder for results and logs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="retrievals.jsonl",
        help="Output file name for the results",
    )
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument(
        "--filter_model",
        type=str,
        default=None,
        help="Model to use for initial filter. If none is specified, then do not perform initial filtering",
    )
    parser.add_argument("--embedding_model", type=str, default="voyage-code-3")
    parser.add_argument(
        "--entire_file", action="store_true", help="Retrieve entire file contents"
    )
    parser.add_argument(
        "--just_create_index",
        action="store_true",
        help="Create the index without performing retrieval",
    )
    parser.add_argument(
        "--enable_weave", action="store_true", help="Enable weave initialization"
    )
    parser.add_argument(
        "--filter_python", action="store_true", help="Filter out non-python files"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.testbed_dir = os.path.join("testbed")
    os.makedirs(args.testbed_dir, exist_ok=True)
    if args.enable_weave:
        weave.init(f"agentless_{args.output_folder}")

    retrieve_swe(args)
