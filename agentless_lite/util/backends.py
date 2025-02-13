import json
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy

import openai

from agentless_lite.util.logging import setup_logging
from agentless_lite.util.repair import create_diff_from_response

STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": STR_REPLACE_EDITOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
                "additionalProperties": False,
            },
        },
    }
]

DEEPSEEK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": STR_REPLACE_EDITOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
            },
        },
    }
]


class CodeGenerator(ABC):
    @abstractmethod
    def generate(self, instance, prompt, args, file_lock, output_file):
        pass

    @abstractmethod
    def initialize_output_files(self, args):
        pass


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(
        base_url=base_url,
    )

    while ret is None and retries < max_retries:
        try:
            logger.info("Creating API request")
            ret = client.chat.completions.create(**config)

            if ret is None or not hasattr(ret, "choices"):
                logger.error(f"Invalid response received: {ret}")
                raise Exception("Invalid API response")

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.error("Request invalid")
                logger.error(str(e))
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                logger.info("Rate limit exceeded. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                logger.info("API connection error. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            elif isinstance(
                e, openai.APITimeoutError
            ):  # Add specific handling for timeout
                logger.info(f"Request timed out after {timeout} seconds. Retrying...")
                logger.error(str(e))
                time.sleep(1)
            else:
                logger.info("Unknown error. Waiting...")
                logger.error(str(e))
                time.sleep(1)

        retries += 1
        if retries >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded")
            ret = None

    logger.info(f"API response {ret}")
    return ret


class BaseGenerator(CodeGenerator):
    def get_temperature_for_generation(self, gen_index, max_temp, total_gens):
        if not hasattr(self, "temperature_list"):
            temp_list = [0.0]  # First generation at 0

            for temp in [0.2, 0.4, 0.6, 0.8]:
                temp_list.extend([temp] * 4)

            for temp in [1.0, 1.2]:
                temp_list.extend([temp] * 8)

            current_temp = 1.4
            while len(temp_list) < total_gens:
                temp_to_add = min(current_temp, max_temp)
                temp_list.extend([temp_to_add] * 8)
                current_temp += 0.2

            temp_list = temp_list[:total_gens]
            self.temperature_list = temp_list

        return self.temperature_list[gen_index]

    def initialize_output_files(self, args):
        if not os.path.exists(args.output_file):
            with open(args.output_file, "w", encoding="utf-8") as outfile:
                pass

    def get_existing_entry(self, instance, file_lock, output_file):
        with file_lock:
            with open(output_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        entry = json.loads(line.strip())
                        if entry["instance_id"] == instance["instance_id"]:
                            return entry
                    except json.JSONDecodeError:
                        continue
        return None

    def update_output_file(self, output_entry, instance, file_lock, output_file):
        with file_lock:
            with open(output_file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            entry_updated = False
            for j, line in enumerate(lines):
                try:
                    entry = json.loads(line.strip())
                    if entry["instance_id"] == instance["instance_id"]:
                        lines[j] = json.dumps(output_entry) + "\n"
                        entry_updated = True
                        break
                except json.JSONDecodeError:
                    continue

            if not entry_updated:
                lines.append(json.dumps(output_entry) + "\n")

            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines)

    def generate_with_retries(self, instance, prompt, args, file_lock, output_file):
        # Create a thread-local copy of args
        local_args = deepcopy(args)

        # Get logs directory relative to output file
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)

        temperature = local_args.temp
        local_args.max_retries = 1

        for attempt in range(args.max_retries):
            try:
                local_args.temp = temperature
                response, output_entry = self.generate(
                    instance,
                    prompt,
                    local_args,
                    file_lock,
                    output_file,
                    defer_writing=True,
                )
                git_diff = create_diff_from_response(
                    response, instance["file_contents"], instance["found_files"]
                )

                if git_diff:
                    # Write both the generation output and the diff
                    with file_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            output = {
                                "instance_id": instance["instance_id"],
                                "model_name_or_path": "agentless_lite_greedy",
                                "model_patch": git_diff,
                                "temperature": temperature,
                                "attempt": attempt + 1,
                            }
                            f.write(json.dumps(output) + "\n")

                    return True

                temperature = min(1.0, temperature + 0.1)
            except Exception as e:
                logger.error(f"Error in generation attempt {attempt + 1}: {e}")
                temperature = min(1.0, temperature + 0.1)

        return False


class OpenAIGenerator(BaseGenerator):
    def generate(
        self, instance, prompt, args, file_lock, output_file, defer_writing=False
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Initializing OpenAI client")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        start_index = len(all_responses)

        if args.warming:
            # Group generations by temperature
            temp_groups = {}
            for i in range(start_index, args.max_retries):
                temperature = self.get_temperature_for_generation(
                    i, args.temp, args.max_retries
                )
                if temperature not in temp_groups:
                    temp_groups[temperature] = []
                temp_groups[temperature].append(i)

            # Generate samples for each temperature group
            for temperature, indices in temp_groups.items():
                num_samples = len(indices)
                logger.info(
                    f"Making API call for {num_samples} samples with temperature {temperature}"
                )

                if args.model == "o3-mini":
                    config = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "n": args.max_retries - start_index,
                        "max_completion_tokens": args.max_completion_tokens,
                        "response_format": {"type": "text"},
                        "reasoning_effort": "high",
                    }
                else:
                    config = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "n": args.max_retries - start_index,
                        "temperature": temperature,
                        "max_tokens": args.max_completion_tokens,
                        "logprobs": args.logprobs,
                    }

                completion = request_chatgpt_engine(config, logger)

                if completion is None:
                    raise Exception("Failed to get response from API")

                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in completion.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].extend([logprobs_data] * num_samples)

                all_responses.extend(
                    [choice.message.content for choice in completion.choices]
                )

                # Update usage information
                all_usage["completion_tokens"] += completion.usage.completion_tokens
                all_usage["prompt_tokens"] += completion.usage.prompt_tokens
                all_usage["temp"].extend([temperature] * num_samples)
                if args.logprobs and hasattr(completion, "logprobs"):
                    all_usage["logprobs"].extend([completion.logprobs] * num_samples)

                output_entry = {
                    "instance_id": instance["instance_id"],
                    "found_files": instance["found_files"],
                    "file_contents": instance["file_contents"],
                    "responses": all_responses,
                    "usage": all_usage,
                }

                if not defer_writing:
                    self.update_output_file(
                        output_entry, instance, file_lock, output_file
                    )
        else:
            # Generate all samples in one batch
            temperature = args.temp
            logger.info(f"Making batch API call with temperature {temperature}")

            if args.model == "o3-mini":
                config = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": args.max_retries - start_index,
                    "max_completion_tokens": args.max_completion_tokens,
                    "response_format": {"type": "text"},
                    "reasoning_effort": "high",
                }
            else:
                config = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": args.max_retries - start_index,
                    "temperature": temperature,
                    "max_tokens": args.max_completion_tokens,
                    "logprobs": args.logprobs,
                }

            completion = request_chatgpt_engine(config, logger)

            if completion is None:
                raise Exception("Failed to get response from API")

            if args.logprobs:
                logprobs_data = [
                    {
                        "token": lp.token,
                        "logprob": lp.logprob,
                        "bytes": lp.bytes,
                        "top_logprobs": lp.top_logprobs,
                    }
                    for lp in completion.choices[0].logprobs.content
                ]
                all_usage["logprobs"].extend(
                    [logprobs_data] * (args.max_retries - start_index)
                )

            all_responses.extend(
                [choice.message.content for choice in completion.choices]
            )

            all_usage["completion_tokens"] += completion.usage.completion_tokens
            all_usage["prompt_tokens"] += completion.usage.prompt_tokens
            all_usage["temp"].extend([temperature] * (args.max_retries - start_index))
            if args.logprobs and hasattr(completion, "logprobs"):
                all_usage["logprobs"].extend(completion.logprobs)

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


class DeepSeekGenerator(BaseGenerator):
    def generate(
        self, instance, prompt, args, file_lock, output_file, defer_writing=False
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Preparing DeepSeek configuration")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        # Make multiple API calls based on max_retries
        start_index = len(all_responses)
        for i in range(start_index, args.max_retries):
            temperature = (
                self.get_temperature_for_generation(i, args.temp, args.max_retries)
                if args.warming
                else args.temp
            )
            logger.info(
                f"Making API call {i+1}/{args.max_retries} with temperature {temperature}"
            )

            if args.model == "deepseek-reasoner":
                config = {
                    "model": args.model,
                    "max_tokens": args.max_completion_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                config = {
                    "model": args.model,
                    "max_tokens": args.max_completion_tokens,
                    "temperature": temperature,
                    "n": 1,
                    "logprobs": args.logprobs,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                }

            if args.tool_use:
                config.update(
                    {
                        "tools": DEEPSEEK_TOOLS,
                        "tool_choice": "required",
                    }
                )

            gener = request_chatgpt_engine(
                config, logger=logger, base_url="https://api.deepseek.com"
            )

            if gener:
                response = (
                    gener.choices[0].message.tool_calls[0]
                    if args.tool_use
                    else gener.choices[0].message.content
                )
                all_responses.append(response)
                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in gener.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].append(logprobs_data)
                if args.warming:
                    all_usage["temp"].append(temperature)

            else:
                all_responses.append("")

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


class OpenRouterGenerator(BaseGenerator):
    def generate(
        self, instance, prompt, args, file_lock, output_file, defer_writing=False
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Preparing OpenRouter configuration")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        # Make multiple API calls based on max_retries
        start_index = len(all_responses)
        for i in range(start_index, args.max_retries):
            temperature = (
                self.get_temperature_for_generation(i, args.temp, args.max_retries)
                if args.warming
                else args.temp
            )
            logger.info(
                f"Making API call {i+1}/{args.max_retries} with temperature {temperature}"
            )

            config = {
                "model": args.model,
                "temperature": temperature,
                "n": 1,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }

            if args.tool_use:
                config.update(
                    {
                        "tools": DEEPSEEK_TOOLS,
                        "tool_choice": "required",
                    }
                )

            gener = request_chatgpt_engine(
                config, logger=logger, base_url="https://openrouter.ai/api/v1"
            )

            if gener:
                response = (
                    gener.choices[0].message.tool_calls[0]
                    if args.tool_use
                    else gener.choices[0].message.content
                )
                all_responses.append(response)
            else:
                all_responses.append("")
                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in gener.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].append(logprobs_data)
                if args.warming:
                    all_usage["temp"].append(temperature)

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


def get_generator(backend_type):
    generators = {
        "openai": OpenAIGenerator(),
        "deepseek": DeepSeekGenerator(),
        "open_router": OpenRouterGenerator(),
    }
    return generators.get(backend_type)
