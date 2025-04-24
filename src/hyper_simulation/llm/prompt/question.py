from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate

question_to_declarative_sentence_template = """You are a helpful assistant of a editor.
You need to translate the question into a declarative answer, with the result being asked marked with {marking}.

-Example- 1:
Question:
Who served as Eisenhower's vice president?
Mark the answer with #1.
Sentence:
Eisenhower's vice president was #1.
-Example- 2:
Question:
Mark the answer with #2.
#1 was a president of what country?
Sentence:
#1 was a president of #2.
-Example- 3:
Question:
Along with the #2 , what major power recognized Gaddafi's government at an early date?
Mark the answer with #3.
Sentence:
Along with the #2 , #3 recognized Gaddafi's government at an early date.

-Real data-
Question:
{question}
Mark the answer with {marking}.
Sentence:
"""
question_to_declarative_sentence = PromptTemplate(
    input_variables=["question", "marking"],
    template=question_to_declarative_sentence_template,
    partial_variables={},
    validate_template=True,
)

sentence_to_subgraph_template = """You are a helpful assistant that could convert a declarative sentence to a graph.
The graph needs to be able to restore all the information in the sentence.

-Step-
1. Identify the nodes and their types in the sentence, which is type of [{entity_types}].
Format: ("node", "<name>", "<type>", "<description>");
2. Identify the edges between the nodes, which is type of [{relation_types}].
Format: ("edge", "<source>", "<target>", "<type>", "<description>");
3. Identify the attributes of the nodes and edges.
Format: ("attribute", "<key>", "<value>", "<desc>", "<node>");
4. When finished, output <END_OUTPUT>

-Examples-
Example 1:
Sentence:
Eisenhower's vice president was #1.
Output:
("node", "Eisenhower", "Person", "The 34th president of the United States.");
("node", "#1", "Person", "Eisenhower's vice president.");
("edge", "Eisenhower", "#1", "Fact", "Eisenhower's vice president is #1.");
<END_OUTPUT>
Example 2:
Sentence:
#1 was a president of #2.
Output:
("node", "#1", "Person", "A president.");
("node", "#2", "Organization", "A country.");
("edge", "#1", "#2", "Fact", "#1 was a president of #2.");
<END_OUTPUT>
Example 3:
Question:
Along with the #2 , #3 recognized Gaddafi's government at an early date.
Output:
("node", "#2", "Organization", "A country.");
("node", "#3", "Organization", "A country.");
("node", "Gaddafi's government", "Organization", "Gaddafi's government.");
("edge", "#2", "Gaddafi's government", "Action", "#2 recognized Gaddafi's government at an early date.");
("edge", "#3", "Gaddafi's government", "Fact", "#3 recognized Gaddafi's government at an early date.");
<END_OUTPUT>

-Real data-
Sentence:
{sentence}
Output:
"""
sentence_to_subgraph = PromptTemplate(
    input_variables=["sentence", "entity_types", "relation_types"],
    template=sentence_to_subgraph_template,
    partial_variables={},
    validate_template=True,
)

entities_numbering_template = """You are a helpful assistant that could recognize all the entities in a text.
You need to number the entities in the text, and the numbering should be consistent with the text.
[Hint]: Pronouns present in the text should also be labeled with the corresponding number based on contextual information.

-Step-
1. Identify the entities in the text, each gets a number.
Format: (<number>, <description>);
2. Return the text with the entities numbered.
Format: [<text>];

-Examples-
Example 1:
Text: Apple Inc. announced a new product launch in New York on March 15, 2025. The CEO, Tim Cook, stated that he expects it to revolutionize the tech market. The event will be hosted at the Hudson Yards Convention Center, a landmark location.
Output:
(#1, "Apple Inc.");
(#2, "New York");
(#3, "March 15, 2025");
(#4, "Tim Cook, the CEO");
(#5, "the tech market");
(#6, "the Hudson Yards Convention Center");
(#7, "a new product");
(#8, "the event of [#1, #2, #3, #4, #5, #7]");  
[#1 announced a #7 launch in #2 on #3. #4, stated that #4 expects #7 to revolutionize the #5. The event will be hosted at #6, a landmark location.]
"""