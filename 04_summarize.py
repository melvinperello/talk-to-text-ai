import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, dtype=torch.float16
).cuda()

# eval_res = model.eval()


def count_tokens(text):
    # Encode the text into tokens
    tokens = tokenizer(
        text,
        return_tensors="pt",
    )
    # Return the number of tokens
    return len(tokens["input_ids"][0])


def chunk_text(text, max_tokens=800):
    """Split text into chunks under the model's token limit."""
    tokens = tokenizer(text, return_tensors="pt", truncation=False)[
        "input_ids"
    ][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks


prompt_0 = """Summarize the following into a concise paragraph that captures the main points, 
key facts, and overall context. Avoid unnecessary details or repetition. 
Focus on clarity, accuracy, and readability."""

# For General Semantic Search
prompt_1 = """Summarize the following text into 2–3 sentences highlighting the main topic, 
important entities, and key facts. The summary should be self-contained 
and optimized for embedding in a vector store.
"""

# For QA / Knowledge Base Retrieval
prompt_2 = """Summarize the following text into a short, factual paragraph that preserves names, dates, numbers, and technical terms.
Focus on accuracy and context so the summary can be used for answering questions later.
"""

# For Thematic Clustering
prompt_3 = """Summarize the following text into a single sentence that captures the 
central theme or subject. Use clear, descriptive language suitable for 
topic clustering in a vector database.
"""


def summarize(input_text):
    inputs = tokenizer(
        f"""INSTRUCTIONS:
{prompt_2}

TEXT:\n\n{input_text}""",
        return_tensors="pt",
        max_length=1024,  # BART limit
        truncation=True,
    ).to("cuda")

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=400,  # allow longer summaries
            min_length=150,
            num_beams=8,  # more thorough exploration
            length_penalty=0.8,  # allows longer outputs
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


text = """

"""

print("Token count:", count_tokens(text))


# print("------ SUMMARIZING ------")
# print(summarize(text))

chunks = chunk_text(text)

for chunk in chunks:
    summary = summarize(chunk)
    print("------ CHUNK SUMMARY ------")
    print("Input Token Count:", count_tokens(chunk))
    print(summary)
    print("Summary Token Count:", count_tokens(summary))
