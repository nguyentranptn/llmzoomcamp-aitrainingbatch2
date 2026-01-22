import os
import json
from tqdm import tqdm
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate


def build_openai_evaluator(
    model_name: str,
    openai_api_key: str,
    base_url: str,
    temperature: float = 0.0,
):
    """Builds an OpenAI chat model used as evaluator."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key,
        base_url=base_url
    )


def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
    max_eval: int = 5,  # add limit (only for test)
) -> None:
    """
    Evaluates generated answers using an LLM-as-a-Judge approach.

    The function modifies the answer file in place to allow checkpointing.
    """
    if not os.path.isfile(answer_path):
        raise FileNotFoundError(f"Answer file not found: {answer_path}")

    with open(answer_path, "r", encoding="utf-8") as f:
        answers = json.load(f)

    evaluated_count = 0

    for experiment in tqdm(answers, desc=f"Evaluating with {evaluator_name}"):
        if evaluated_count >= max_eval:
            break  # stop after max_eval

        score_key = f"eval_score_{evaluator_name}"
        feedback_key = f"eval_feedback_{evaluator_name}"

        # Skip if already evaluated
        if score_key in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )

        eval_result = eval_chat_model.invoke(eval_prompt)

        try:
            feedback, score = [
                item.strip()
                for item in eval_result.content.split("[RESULT]")
            ]
        except ValueError:
            raise ValueError(
                f"Invalid evaluator output format:\n{eval_result.content}"
            )

        experiment[score_key] = int(score)
        experiment[feedback_key] = feedback

        evaluated_count += 1  # increment counter

        # Save checkpoint after each evaluation
        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)

