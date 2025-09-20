# RAG_with_llama3_1

Start venv
```bash
source ~/venvs/ai_test/bin/activate
```


# RAG experiment — step-by-step explanation

This Markdown file explains your simple RAG (Retrieval-Augmented Generation) experiment line-by-line and gives practical tips, troubleshooting, and next steps.

---

## Overview

You built a small RAG pipeline that:

1. Loads a local text file (`my_knowledge.txt`) as the knowledge source.
2. Splits the document into chunks.
3. Converts chunks into embeddings using a sentence-transformer model.
4. Stores embeddings in a FAISS vector index.
5. Builds a retriever and composes a RAG chain with a local LLM (Llama 3.1 8B Instruct) running via Hugging Face's pipeline and LangChain.
6. Queries the chain and returns answers constrained to the retrieved context.

---

## Prerequisites

Make sure you have these Python packages (example install commands):

```bash
pip install torch transformers bitsandbytes sentence-transformers faiss-cpu huggingface-hub langchain langchain-community
```

Notes:

* If you have a CUDA GPU, install the GPU builds of `torch` and `faiss` appropriate for your CUDA version.
* `bitsandbytes` requires a compatible GPU and CUDA setup for efficient 4-bit quantization.
* The script also uses `huggingface_hub.login()` if you want to access private HF models.

---

## Step 0 — Prepare your knowledge file

You created `my_knowledge.txt`. here it is with the small Aethelred OS description. This is the knowledge base that RAG will search.

```python
%%writefile my_knowledge.txt
The Aethelred Operating System, first released in 2023, ...
```

This is a single-file, single-document knowledge source — fine for an experiment. For production, you typically work with many documents (PDFs, webpages, CSVs, etc.).

---

## Step 1 — Load the document

You used a `TextLoader` from `langchain_community.document_loaders`:

```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("./my_knowledge.txt")
documents = loader.load()
```

**What this does:** reads the file and returns one or more `Document` objects (text + optional metadata).

---

## Step 2 — Split the document into chunks

You split the document to keep each retrieved chunk under the model's context window and to give the retriever manageable pieces:

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks.")
```

**Why chunk & overlap?**

* `chunk_size` controls the maximum characters per piece; `chunk_overlap` keeps related context across chunks.
* Overlap helps avoid cutting an important sentence in half so that relevant context persists across nearby chunks.
* Choose chunk size based on (model context length) and the typical length of coherent passages in your documents.

---

## Step 3 — Create embeddings

You used a compact sentence-transformer:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
```

**Notes on model choice:**

* `all-MiniLM-L6-v2` is small and fast, good for prototyping and local embedding generation.
* For higher semantic accuracy, consider larger models (e.g., `all-mpnet-base-v2` or OpenAI embeddings if accuracy matters and the budget allows).
* Running embeddings locally means you don't need an external API key, but you need enough RAM/CPU.

---

## Step 4 — Build the FAISS vector store

```python
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(chunks, embedding_model)
print("Vector store created successfully.")
```

**What FAISS does:** stores vectors and performs fast nearest-neighbor lookups. The `from_documents` call creates embeddings for each chunk and builds the index.

**Persistence:**

* To reuse the index later without re-embedding, save it to disk. LangChain/FAISS wrappers usually provide `save_local(...)` and `load_local(...)` methods (check your wrapper's API). Persisting saves time.

---

## Step 5 — (Optional) Hugging Face login / HF token

You read the HF token from env and commented out `login("YOUR_HF_TOKEN")`.

```python
from huggingface_hub import login
import os
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
# login("YOUR_HF_TOKEN")
```

If you need to download gated HF models, log in or set `HF_HOME` credentials.

---

## Step 6 — Clean-up helper function

You define `clean_answer` which strips a custom header token from model output:

```python
def clean_answer(raw_output: str) -> str:
    if "assistant<|end_header_id|>" in raw_output:
        return raw_output.split("assistant<|end_header_id|>")[-1].strip()
    return raw_output.strip()
```

**Why:** your prompt/template uses header tokens like `assistant<|end_header_id|>` — the LLM output may include them. This helper removes them so the printed answer is neat.

---

## Step 7 — Load the LLM with quantization (BNB)

You load a Llama 3.1 8B instruct model and use 4-bit quantization for memory savings:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Key points:**

* `load_in_4bit=True` drastically reduces GPU memory but requires `bitsandbytes` support.
* `nf4` quant type is a modern choice that often preserves more quality in extreme quantization.
* `bnb_4bit_compute_dtype=torch.bfloat16` sets internal compute dtype — `bfloat16` might not be available on all GPUs; you may use `torch.float16` or `torch.float32` as fallback.
* `device_map="auto"` lets Accelerate/transformers place model shards across devices.

**Warnings:**

* If your hardware doesn't support bfloat16 or 4-bit acceleration, you may hit runtime errors or suboptimal performance.
* For CPU-only environments, quantized GPU-only features might not work.

---

## Step 8 — Create a text-generation pipeline and wrap it for LangChain

```python
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    top_p=0.9,
    temperature=0.1,
)

from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

**Parameters explained:**

* `max_new_tokens=256`: how many tokens the model can generate.
* `top_p=0.9`: nucleus sampling — you let the model sample from the top 90% probability mass.
* `temperature=0.1`: low temperature makes outputs more deterministic (good when you want fact-like answers from a RAG flow).

For the RAG use-case, lower `temperature` + prompt instructions to only use the provided context tends to produce more reliable answers.

---

## Step 9 — Prompt template

You created a strict prompt instructing the model to only use context and to reply "I don't know" when the answer isn't in context:

```python
from langchain.prompts import PromptTemplate

prompt_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Please answer the user's question based only on the following context. If the answer is not in the context, say you don't know. Do not use any prior knowledge.

CONTEXT:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
```

**Why this matters:**

* You force the LLM to ground its response in the retrieved `context` rather than free-roaming on internet knowledge.
* Be mindful: models can still hallucinate. Binding a model *perfectly* to only use retrieved context is an ongoing research challenge.
* The special tokens (`<|...|>`) are just separators; they must match what your LLM/decoder expects. Some LLMs ignore them; treat them as guidance.

---

## Step 10 — Create the retriever and RAG chain

Retriever:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

* `k=2` means "retrieve the top 2 most similar chunks". Increase `k` if context is too small or if you want more evidence; decrease `k` for faster, narrower context.

RAG chain (LangChain Expression Language):

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**Flow explanation:**

1. The retriever receives the incoming question and returns the `context` (top-k chunks).
2. `prompt` formats `context` and `question` into the text the LLM expects.
3. `llm` generates an answer.
4. `StrOutputParser()` converts the raw model output into a string you can use.

Note: This LCEL pipeline composes a retriever and the prompt and LLM into a single callable object.

---

## Step 11 — Run example queries

You ran two queries:

```python
question1 = "What is the primary programming language for Aethelred OS?"
answer1 = clean_answer(rag_chain.invoke(question1))

question2 = "Who is the CEO of the company that makes Aethelred OS?"
answer2 = clean_answer(rag_chain.invoke(question2))
```

**Why the results differ:**

* `question1` has information directly in the knowledge file and thus the retriever finds the chunk and the model answers correctly.
* `question2` is not in the document, so with the prompt instruction the model returns "I don't know." This shows the RAG pipeline is behaving as requested.

---

## Step 12 — Runtime notes & warnings

* `Setting pad_token_id to eos_token_id:128001` — typically occurs because the tokenizer has no `pad_token` defined and HF sets it equal to the `eos_token` to avoid errors during generation.
* Quantized and large-model loading messages — `device_map` and `bitsandbytes` will print placement and memory information.

---

## Step 13 — Practical improvements and suggestions

1. **Persistence**: Save the FAISS index and chunk metadata to disk so you don't re-embed every run.
2. **Reranker**: Use an (optional) cross-encoder reranker on the top-N retrieved chunks to choose the best context before generation.
3. **Metadata**: Keep metadata (document id, source, offsets) for each chunk and return/source-cite them in the answer.
4. **Relevance diversity**: Use MMR (maximal marginal relevance) to reduce redundant retrieved chunks.
5. **Chunking strategy**: Use a semantic-aware splitter (e.g., recursive or sentence-based) instead of a raw character splitter for richer context boundaries.
6. **Prompt engineering**: Add an explicit `Sources:` section and require answers to include the chunk id or source lines.
7. **Confidence / Abstain**: Have the LLM output a confidence score or force it to abstain when context is insufficient.
8. **Evaluation**: Create a test set with expected answers and measure exact-match / F1 to evaluate retrieval + generation.

---

## Step 14 — Troubleshooting common errors

* **OOM (out-of-memory)**: reduce model size, use `bnb` quantization (as you did), lower `max_new_tokens`, or run on a GPU with more memory.
* **bitsandbytes errors**: ensure compatible CUDA & `bitsandbytes` builds; if not possible, try `load_in_8bit=True` or run the model in float16/float32.
* **Tokenizer missing pad token**: explicitly call `tokenizer.pad_token = tokenizer.eos_token` before creating the pipeline to silence warnings.
* **Hugging Face login failures**: ensure `HF_TOKEN` is set and you call `huggingface_hub.login(token)` or pass token to `from_pretrained(..., use_auth_token=hf_token)`.
* **Slow retrieval / embedding**: precompute and persist embeddings; use a faster model for embeddings in production.

---
