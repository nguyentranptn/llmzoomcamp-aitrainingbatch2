# Agent

- [Agent](#agent)
  - [What is AI Agent?](#what-is-ai-agent)
  - [LLM Using Tools](#llm-using-tools)
    - [What is a MRKL System?](#what-is-a-mrkl-system)
    - [Examples of MRKL Systems](#examples-of-mrkl-systems)
    - [Components](#components)
    - [Model](#model)

## What is AI Agent?

Agents are LLMs that can use external tools, such as APIs and databases, to incorporate information beyond the model's training and enhance problem-solving abilities.

One of the primary difference between Agents and LLMs is that Agents can **make decision** based on its input and surrounding environments. One example could be generating an answer for a question that is out side of LLM's knowledge. A typical LLM will make up the answer, while an Agents can call tools to perform an Internet search for the information, and then compose the final answer.

Let's take a look at this example of an Agent:

Prompt:

```text
What is 19 percent of 5619?
```

Output:

```text
CALCULATOR[(0.19) * (5619)]
```

Can you spot the reason why this is an output from Agents, not a LLM? Let's discuss your answer in class.

## LLM Using Tools

### What is a MRKL System?

Modular Reasoning, Knowledge, and Language or MRKL Systems (pronounced "miracle") are an architecture that combines Large Language Models (neural computation) with external tools like calculators to solve complex problems.

A MRKL system consists of a set of modules (e.g., a calculator, weather API, database) and a router that decides how to 'route' incoming natural language queries to the appropriate module.

A simple example of a MRKL system is an LLM that can use a calculator app. This is a single-module system, where the LLM acts as the router. When asked, "What is 100 \* 100?", the LLM extracts the numbers from the prompt and directs the MRKL system to use the calculator to compute the result, like this:

Prompt:

```text
What is 100 x 100?
```

Output:

```text
CALCULATOR[100 * 100]
```

### Examples of MRKL Systems

Consider the following additional examples of applications:

- **Financial Database Queries**: A chatbot can extract information from a user's text and form a SQL query to fetch the latest stock prices.
- **Weather Queries**: A chatbot can extract location data from a prompt and use a weather API to provide the current forecast.

### Components

An LLM Agent runs tools in a loop to achieve a goal. An agent runs until a stop condition is met - i.e., when the model emits a final output or an iteration limit is reached.

Below is the workflow of this process, taken from LangChain:

![Agent Workflow](agent-workflow.png)

### Model

The model is the reasoning engine of your agent. In LangChain can be specified in multiple ways, supporting both static and dynamic model selection.
