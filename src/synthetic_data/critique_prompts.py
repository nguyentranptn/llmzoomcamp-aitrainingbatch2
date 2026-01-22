QA_GENERATION_PROMPT = """
Your task is to write a factoid question and an answer given a context.

Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.

This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context:
{context}

Output:::
"""

QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5...

Question: {question}
Context: {context}
Answer:::

Do not include anything else.
"""

QUESTION_RELEVANCE_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be...

Question: {question}
Answer:::

Do not include anything else.
"""

QUESTION_STANDALONE_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is...

Question: {question}
Answer:::

Do not include anything else.
"""
