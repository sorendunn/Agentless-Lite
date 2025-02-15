import os

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding


def get_embedding_model(model_name, logger):
    """Get the appropriate embedding model based on the model name.

    Args:
        model_name (str): Name of the embedding model
        logger: Logger instance for logging information

    Returns:
        The initialized embedding model

    Raises:
        Exception: If the model name is not supported
    """
    if model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
        model = OpenAIEmbedding(model_name=model_name, embed_batch_size=128)
    elif "voyage" in model_name:
        model = VoyageEmbedding(
            model_name=model_name,
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            embed_batch_size=128,
            output_dimension=2048,
        )
    else:
        logger.error(f"Unsupported embedding model: {model_name}")
        raise Exception("Unsupported embedding model")
    return model


def create_index_with_retry(documents, embedding_model, save_dir, logger):
    """Attempts to create an index with decreasing batch sizes on failure.

    Args:
        documents (list): List of Document objects
        embedding_model: The embedding model to use
        save_dir (str): Directory to save the index
        logger: Logger instance for logging information

    Returns:
        VectorStoreIndex: The created index

    Raises:
        Exception: If index creation fails with all batch sizes
    """
    batch_sizes = [128, 64, 1]

    for batch_size in batch_sizes:
        try:
            embedding_model.embed_batch_size = batch_size
            logger.info(f"Attempting to create index with batch size {batch_size}")
            index = VectorStoreIndex.from_documents(
                documents, embed_model=embedding_model
            )
            index.storage_context.persist(persist_dir=save_dir)
            return index
        except Exception as e:
            logger.warning(
                f"Failed to create index with batch size {batch_size}: {str(e)}"
            )
            if batch_size == 1:
                logger.error("Failed to create index with minimum batch size")
                raise


def create_new_index(files, embedding_model, save_dir, file_to_contents, logger):
    documents = []
    for file in files:
        file_content = file_to_contents[file]
        meta_data = {
            "File Name": file,
        }
        doc = Document(
            text=file_content,
            metadata=meta_data,
            metadata_template="{key}: {value}",
            text_template="{metadata_str}\n-----\nCode:\n{content}",
        )
        documents.append(doc)
    return create_index_with_retry(documents, embedding_model, save_dir, logger)


def retrieve(
    files,
    prompt,
    embedding_model_name,
    save_dir,
    filter_num,
    retrieve_num,
    file_to_contents,
    entire_file,
    just_create_index=False,
    logger=None,
    filter_model_name=None,
):

    original_file_to_contents = file_to_contents

    embedding_model = get_embedding_model(embedding_model_name, logger)

    if filter_model_name:
        filter_save_dir = save_dir + "_filter"
        filter_model = get_embedding_model(filter_model_name, logger)

        Settings.embed_model = filter_model
        if (
            os.path.exists(save_dir)
            and os.path.isdir(save_dir)
            and os.listdir(save_dir)
        ):
            logger.info("Main index exists, skipping filter index creation")
            filter_index = None
        else:
            if (
                os.path.exists(filter_save_dir)
                and os.path.isdir(filter_save_dir)
                and os.listdir(filter_save_dir)
            ):
                try:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=filter_save_dir
                    )
                    filter_index = load_index_from_storage(storage_context)
                    logger.info("Loading existing filter index from storage")
                except Exception as e:
                    logger.error(f"Failed to load filter index from storage: {str(e)}")
                    logger.info("Recreating filter index")
                    filter_index = create_new_index(
                        files, filter_model, filter_save_dir, file_to_contents, logger
                    )
            else:
                logger.info("Creating new filter index")
                filter_index = create_new_index(
                    files, filter_model, filter_save_dir, file_to_contents, logger
                )

            filter_retriever = VectorIndexRetriever(
                index=filter_index,
                embed_model=filter_model,
                similarity_top_k=filter_num,
            )
            filter_docs = filter_retriever.retrieve(prompt)
            logger.info(
                f"Retrieved {len(filter_docs)} sections using {filter_model.model_name}"
            )

            filtered_contents = {}
            for node in filter_docs:
                file_name = node.metadata["File Name"]
                if file_name not in filtered_contents:
                    filtered_contents[file_name] = []
                filtered_contents[file_name].append(node.text)

            final_filtered_contents = {}
            for file_name, sections in filtered_contents.items():
                final_filtered_contents[file_name] = "\n...\n".join(sections)

            file_to_contents = final_filtered_contents

    Settings.embed_model = embedding_model
    if os.path.exists(save_dir) and os.path.isdir(save_dir) and os.listdir(save_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=save_dir)
            index = load_index_from_storage(storage_context)
            logger.info("Loading existing index from storage")
        except Exception as e:
            logger.error(f"Failed to load index from storage: {str(e)}")
            logger.info("Recreating index")
            index = create_new_index(
                file_to_contents.keys(),
                embedding_model,
                save_dir,
                file_to_contents,
                logger,
            )
    else:
        logger.info("Creating new index")
        index = create_new_index(
            file_to_contents.keys(), embedding_model, save_dir, file_to_contents, logger
        )

    if just_create_index:
        return [], []
    else:
        retriever = VectorIndexRetriever(
            index=index, embed_model=embedding_model, similarity_top_k=retrieve_num
        )
        retrieved_documents = retriever.retrieve(prompt)
        logger.info(
            f"Retrieved {len(retrieved_documents)} sections using {embedding_model.model_name}"
        )

        file_names = []
        file_contents = []
        for node in retrieved_documents:
            file = node.metadata["File Name"]
            if not entire_file:
                file_names.append(file)
                file_contents.append(node.text)
            else:
                if file not in file_names:
                    file_names.append(file)
                    file_contents.append(original_file_to_contents[file])

        return file_names, file_contents
