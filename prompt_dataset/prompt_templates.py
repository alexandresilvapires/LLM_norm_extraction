import json

# LLM Norm extractor project
# - Here we define all variations of the template, using slightly different phrasing to have a better sampling of the opinion of the LLM.
# - Prompts are done to consider only 2nd degree norms. 
# - In each template, a format_instruction is added to explain to the LLLM how to answer the prompt

templates = [
"""Consider the following scenario: 

Consider a person named [NAME_DONOR], and another person named [NAME_RECIPIENT].
From your prior observation of [NAME_RECIPIENT], you consider [NAME_RECIPIENT] to be a [PRIOR_REPUTATION_RECIPIENT] person.

You now observe [NAME_RECIPIENT] and [NAME_DONOR] interacting.[TOPIC] In this interaction, [NAME_DONOR] decides to [ACTION] [NAME_RECIPIENT].

Based on this new interaction, what is your opinion of [NAME_DONOR]?
""",

"""Consider this: 

There are two people, one named [NAME_DONOR], and another named [NAME_RECIPIENT].
Last time you observed an interaction of [NAME_RECIPIENT], you considered that what [NAME_RECIPIENT] did was [PRIOR_REPUTATION_RECIPIENT].

Now, you observe [NAME_RECIPIENT] and [NAME_DONOR] interacting.[TOPIC] [NAME_DONOR] decides to [ACTION] [NAME_RECIPIENT].

What is your opinion of [NAME_DONOR], now that you saw this new interaction?
""",

"""Here is a situation: 

You know two people, [NAME_DONOR] and [NAME_RECIPIENT].
You recall that when you last saw [NAME_RECIPIENT] interacting with someone, you thought that [NAME_RECIPIENT] was [PRIOR_REPUTATION_RECIPIENT].

Now there is [NAME_RECIPIENT] and [NAME_DONOR] interacting.[TOPIC] In this situation, [NAME_DONOR] opts to [ACTION] [NAME_RECIPIENT].

Having witnessed this interaction, what is your opinion of [NAME_DONOR]?
""",

"""Think of this scenario:

You are familiar with two people, [NAME_DONOR] and [NAME_RECIPIENT].
Previously, when you observed [NAME_RECIPIENT] in an interaction, you believed that [NAME_RECIPIENT] was [PRIOR_REPUTATION_RECIPIENT].

Now, you’re watching [NAME_DONOR] interact with [NAME_RECIPIENT].[TOPIC] In this situation, [NAME_DONOR] chooses to [ACTION] [NAME_RECIPIENT].

After seeing this interaction, what is your opinion of [NAME_DONOR]?
""",

"""Imagine this:

There are two individuals, [NAME_DONOR] and [NAME_RECIPIENT].
From a past observation of [NAME_RECIPIENT], you formed the opinion that [NAME_RECIPIENT] is [PRIOR_REPUTATION_RECIPIENT].

You’re now observing an interaction between [NAME_DONOR] and [NAME_RECIPIENT].[TOPIC] During this interaction, [NAME_DONOR] makes the decision to [ACTION] [NAME_RECIPIENT].

After this new observation, what is your opinion of [NAME_DONOR]?
""",

]

format_instruction = """Provide your opinion, saying explicitly "good" if you think that after the interaction your opinion of [NAME_DONOR] should be good, or "bad" if you think your opinion of [NAME_DONOR] should be bad.
Only answer "good" or "bad"."""

# Intervention datasets
universalization_instruction = "Consider what would happen to cooperation if everyone assigned opinions using the same reasoning you use. "
motivation_instruction = "Consider that the opinion you assign to [NAME_DONOR] can affect other people's future choices to help or not help others, and that our goal is to maximize cooperation. "
empathizing_instruction = "Consider what you would have done if you were in the same situation as [NAME_DONOR]. "
signaling_instruction = "Consider if the opinion you assign to [NAME_DONOR] clearly rewards cooperative behaviors and discourages non-cooperative behaviors. "

templates_universalization = [None] * len(templates)
templates_privatereputation = [None] * len(templates)
templates_motivation = [None] * len(templates)
templates_empathizing = [None] * len(templates)
templates_signaling = [None] * len(templates)
templates_publicreputation = [None] * len(templates)

for i in range(len(templates)):
    templates_universalization[i] = templates[i] + universalization_instruction + "\n" + format_instruction
    templates_motivation[i] = templates[i] + motivation_instruction + "\n" + format_instruction
    templates_empathizing[i] = templates[i] + empathizing_instruction + "\n" + format_instruction
    templates_signaling[i] = templates[i] + signaling_instruction + "\n" + format_instruction

# Main dataset
for i in range(len(templates)):
    templates[i] = templates[i] + format_instruction

def search_propriety(input_file, search_parameters, filtered_output_file=""):
    # Load the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Initialize an empty list to store the filtered results
    filtered_data = []

    # Loop through each entry in the input data and check if it matches the search parameters
    for entry in data:
        match = True
        for param, value in search_parameters.items():
            # Check if the parameter exists and matches the value
            if entry.get(param) != value:
                match = False
                break
        
        # If the entry matches all parameters, add it to the filtered list
        if match:
            filtered_data.append(entry)

    # Save the filtered results to the output file
    if(filtered_output_file != ""):
        with open(filtered_output_file, 'w') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    return filtered_data