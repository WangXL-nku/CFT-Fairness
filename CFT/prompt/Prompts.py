# prompts.py
# Contains all task-specific prompt templates for CFT bias audit.

TASK_DESCRIPTIONS = {
    # Biased tasks
    "Blonde Hair": "Blonde Hair",
    "Heavy Makeup": "Heavy Makeup",
    "Age": "Age (young vs. other)",
    "Gender": "Gender",
    "Digit Class": "Digit Class (bird vs. cat)",
    # Unbiased control tasks
    "Black Hair": "Black Hair",
    "Blurry": "Blurry",
}

DEFAULT_TASK = "Blonde Hair"

def get_prompt(task_name: str) -> str:
    """
    Generate the complete bias audit prompt for the given task.
    """
    task_description = TASK_DESCRIPTIONS.get(task_name, task_name)
    task_lower = task_description.lower().split("(")[0].strip()

    return f"""
Role: You are a professional bias detection expert.

Task Definition: Assess whether a semantic concept visualization contains biased semantics. These visualizations consist of multiple input image patches most relevant to the semantic concept, derived from the interpretable concept decomposition of a deep model. The core classification task is "{task_description}" – determining whether a face image has {task_lower}.

A biased concept refers to a semantic concept that is irrelevant to the core prediction target yet significantly influences the model's prediction. Your evaluation must be objective and independent, with no prior assumption that any specific bias exists in the visualization.

Evaluation Process:

1. Concept Semantic Identification
Carefully examine the visualization content and identify all discernible semantic concepts within it.

2. Bias Assessment
For each identified concept, systematically assess whether it constitutes a biased concept by answering the following questions:
- Is the concept itself inherently relevant to the task of predicting "{task_description}"?
- Based on common descriptions of biased concepts, does this concept align with any known irrelevant biased characteristics?

3. Comprehensive Assessment & Bias Labeling
Based on the above analysis, conduct an overall weighting of all concepts.
Form a final conclusion, clearly stating whether the semantic concept visualization as a whole contains bias, and briefly explain the primary reasoning.
Output your comprehensive bias label (e.g., "Biased Present" or "No Significant Bias Detected").

Independent Analysis Guidelines:
- Do not presuppose that specific biases necessarily exist in the visualization. Conduct a systematic review based on the provided description and its relevance to the "{task_description}" prediction task.
- During the concept semantic identification phase, strive for a comprehensive and thorough observation of all discernible concepts.
- During the bias assessment phase, strictly apply logical judgment to each concept based on the questions above.
- The final conclusion should be based on objective evidence, and the reasoning process should be clearly presented in the comprehensive assessment.
"""