RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked; the response should be concise and relevant.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>

<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>

<|assistant|>
"""
