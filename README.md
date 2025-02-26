# 🐱 Agentless Lite

<p align="center">
    <a href="https://arxiv.org/abs/2407.01489"><img src="https://img.shields.io/badge/📑-Arxiv-b31b1b?style=for-the-badge"></a>
    <a href="https://github.com/sorendunn/Agentless-Lite/blob/master/LICENSE"><img src="https://forthebadge.com/images/badges/license-mit.svg" style="height: 28px"></a>
</p>

<p align="center">
    <big><a href="#-news">📢News</a></big> |
    <big><a href="#-about">💡About</a></big> |
    <big><a href="#-setup">⚙️Setup</a></big> |
    <big><a href="#-quickstart">⚡Quickstart</a></big>
</p>
<p align="center">
    <big><a href="#-artifacts">🐈‍⬛Artifacts</a></big> |
    <big><a href="#-acknowledgement">😻Acknowledgement</a></big>
</p>

## 📢 News

- *Febuary 26th, 2025*: Agentless-Lite more than **doubles SOTA** on SWE-bench Multimodal from 12.19% to 25.34% (6x the performance of Agentless) for a fourth of the cost without even requiring a runtime environment!
- *Febuary 13th, 2025*: We just released Agentless-Lite 1.0! Agentless-Lite is the top-performing **RAG-only** scaffold for SWE-bench, increasing RAG performance on the lite subset from 4.33% to 32.33% and costing only $0.21 per instance ($0.12 if using the prepared retrieval contexts)!

## 💡 About

<p align="left">
    <big>Check out the original Agentless implementation here: <a href="https://github.com/OpenAutoCoder/Agentless">🚀 Agentless Repository</a></big>
</p>

**Agentless Lite** is a generalized, lightweight adaptation of the [Agentless](https://github.com/OpenAutoCoder/Agentless) framework for solving software development issues. Specifically, **Agentless Lite** performs the following steps:

1. Use an embedding model to retrieve relevant files from the repository
2. Query the LLM to generate a repair based on the top 5 retrieved files, retrying the generation until the model outputs a valid patch.

Thats it! While simple this approach is competitive with SOTA agents and comes with several key advantages:

- 🔍 Exclusively RAG-based localization
- 💨 No required runtime environment
- 🐍 No python specific language dependencies
- ⚡ Simple, single-prompt inference
- 🤝 Support for over 300 models with *OpenRouter*
- 💰 Costs less than $0.25 per instance

## ⚙️ Setup

First create the environment

```shell
git clone https://github.com/sorendunn/Agentless-Lite.git
cd Agentless-Lite

conda create -n agentless_lite python=3.11
conda activate agentless_lite
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then set up your OpenAI API key, VOYAGE_API_KEY (if using Voyage embeddings), and WANDB_API_KEY (if using weave):

```shell
export OPENAI_API_KEY={openai_key_here}
export VOYAGE_API_KEY={vogage_key_here}
export WANDB_API_KEY={wandb_key_here}
```

## ⚡ Quickstart

### Prerequisites

1. Download and unzip the prepared retrieval contexts for SWE-Bench Lite [swe_bench_lite.zip](https://github.com/sorendunn/Agentless-Lite/releases/download/v0.1.0/agentless_lite_retrievals.zip) or SWE-Bench Verified [swe_bench_verified.zip](https://github.com/sorendunn/Agentless-Lite/releases/download/v0.1.0/agentless_verified_retrievals.zip)
    - Alternatively, see `Localization` section for how to generate your own retrieval contexts
2. Move the jsonl file to the main Agentless Lite directory (or specify the path with `--loc_file`)

### Run

```shell
python agentless_lite/repair.py \
        --base_path agentless_lite \
        --output_folder results \
        --loc_file retrieval.jsonl \
        --temp 0 \
        --model o3-mini \
        --max_completion_tokens 78000 \
        --max_input_tokens 118000 \
        --backend openai \
        --num_threads 16 \
        --max_retries 10 \
        --max_files 5
```

This command will iteratively prompt the model (gradually increasing the temperature) until a valid patch is produced or the `--max_retries` is reached. The complete logs are also saved in `results/logs` It will produce `all_preds.jsonl` that contains the generated patch for each instance_id which you can then directly evaluate with your favorite SWE-bench evaluation method!

> [!TIP]
>
> We currently support OpenRouter, OpenAI, and DeepSeek models. Additionally we support batch submission for compatible OpenAI models. You can change which of these backends to use via the `--backend` parameter (open_router, openai, openai_batch_offline or deepseek)
>
> For example `--backend deepseek`

## 🐈 Localization

Create the embeddings and perform retrieval:

```shell
python agentless_lite/retrieve_swe.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --num_threads 1 \
        --output_folder results \
        --output_file retrieval.jsonl \
        --embedding_folder voyage_lite \
        --embedding_model voyage-code-3 \
        --filter_model text-embedding-3-small \
        --filter_python \
        --entire_file
```

This will split files in the repositories into small chunks for embedding. `--filter_python` specifies to only embed the non-test python files in the repository. `--entire-file` specifies to retrieve the entire file if any chunks within the file are retrieved. `--retrieve_num` indicates the total number of chunks to retrieve.

> [!TIP]
>
> We currently support OpenAI and Voyage embeddings, you can use `--embedding-model` to select the desired embedding model (by default it will use Voyage embeddings)
>
> For example `--embedding-model=openai_small`

> [!TIP]
>
> We use multiple threads (controllable via `--num-threads`) to speed up the Agentless process

## 🐈‍⬛ Artifacts

You can download the complete artifacts of **Agentless Lite** in our [v0.1.0 release](https://github.com/sorendunn/Agentless-Lite/releases/tag/v0.1.0):

- 🐈‍⬛ **source_code.zip**: source code for Agentless Lite
- 🐈‍⬛ **agentless_lite_retrievals.zip**: top retreived files for filtering + Voyage-Code-3 on SWE-bench Lite
- 🐈‍⬛ **agentless_verified_retrievals.zip**: top retreived files for filtering + Voyage-Code-3 on SWE-bench Verified
- 🐈‍⬛ **agentless_lite_run.zip**: complete Agentless Lite run on SWE-bench Lite for o3-mini
- 🐈‍⬛ **agentless_lite_run_verified.zip**: complete Agentless Lite run on SWE-bench Verified for o3-mini

## 😻 Acknowledgement

* [Agentless](https://github.com/OpenAutoCoder/Agentless)
* [SWE-bench](https://www.swebench.com/)
* [Aider](https://github.com/paul-gauthier/aider)
* [SWE-bench-docker](https://github.com/aorwall/SWE-bench-docker)
