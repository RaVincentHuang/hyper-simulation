from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate

_graph_building_info_template = """ Your are a graph generator for GraphRAG, you will receive a piece of text, you need to convert the text into a graph and output it in `json` format.

-Steps-
1. Identify the entities in the text. You need to analyze sentence by sentence to get as many entities in the text as possible.
You must extract the following information:
- Entity name: each entity should give an individual name.
- Ontology type of the entity: you MUST to classify the entity into one of the following categories: [{entity_types}].
- Entity description: Comprehensive description of the entity's attributes and activities.
2. Format each entity into `json` format of:
{{
    "name": "entity_name",
    "type": "entity_type",
    "desc": "description"
}}
3. Identify the relations between the entities. You need to analyze sentence by sentence to get as many relations in the text as possible.
You must extract the following information:
- Source entity: the source entity of the relation.
- Destination entity: the destination entity of the relation.
- Ontology type of the relation: you MUST to classify the relation into one of the following categories: [{relation_types}].
- Relation description: Comprehensive description of the relation's attributes and activities.
4. Format each relation into `json` format of:
{{
    "src": "src_entity",
    "dst": "dst_entity",
    "type": "relation_type",
    "desc": "description"
}}
5. Collect all the entities and relations into a list of `json` format.
"""

# {{
#     "graph_name": "graph_name",
#     "graph_description": "graph_description",
#     "entities": [
#         {{
#             "name": "entity_name",
#             "type": "entity_type",
#             "desc": "description"
#         }},
#     ],
#     "relations": [
#         {{
#             "src": "src_entity",
#             "dst": "dst_entity",
#             "type": "relation_type",
#             "desc": "description"
#         }},
#     ]
# }}

_graph_building_info_without_type_template = """Your are a graph generator for GraphRAG, you will receive a piece of text, YOU NEED TO STRICTLY FOLLOW THE `json` FORMAT OF THE OUTPUT.
-Format-
{{
    "graph_name": "graph_name",
    "graph_description": "graph_description",
    "entities": [
        {{
            "name": "entity_name",
            "desc": "description"
        }},
    ],
    "relations": [
        {{
            "desc": "description"
            "src": "src_entity",
            "dst": "dst_entity",
        }},
    ]
}}
-Steps-
1. Identify the entities in the text. You need to analyze sentence by sentence to get as many entities in the text as possible.
You must extract the following information:
- Entity name: each entity should give an individual name.
- Entity description: Comprehensive description of the entity's attributes and activities.
2. Identify the relations between the entities. You need to analyze sentence by sentence to get as many relations in the text as possible.
You must extract the following information:
- Relation description: Comprehensive description of the relation's attributes and activities.
- Source entity: the source entity of the relation.
- Destination entity: the destination entity of the relation.
3. Collect all the entities and relations into a list of `json` format.
"""

_graph_building_example_list = [
"""
[Input]
title: Henry Feilden
text: Henry Master Feilden (21 February 1818, 5 September 1875) was an English Conservative Party politician. On 16 March 1869, the result of the 1868 general election in the borough of Blackburn was declared null and void, after an election petition had been lodged. The two Conservatives who had been elected, William Henry Hornby and Feilden's father Joseph Feilden. Henry Feilden was elected at the resulting by-election on 31 March 1869, along with William Henry Hornby's son Edward.
prop: occupation
[Output]
{{
    "graph_name": "Henry Master Feilden",
    "graph_description": "The information of an English Conservative Party politician Henry Master Feilden.",
    "entities": [
        {{
            "name": "Henry Master Feilden",
            "type": "Person",
            "desc": "Henry Master Feilden"
        }},
        {{
            "name": "Feilden's Birthday",
            "type": "Time",
            "desc": "21 February 1818, 5 September 1875"
        }},
        {{
            "name": "English",
            "type": "Attribute",
            "desc": "Feilden is an English"
        }},
        {{
            "name": "Conservative Party",
            "type": "Attribute",
            "desc": "Feilden is a member of the Conservative Party"
        }},
        {{
            "name": "Politician",
            "type": "Attribute",
            "desc": "Feilden is a politician"
        }},
        {{
            "name": "1868 General Election",
            "type": "Event",
            "desc": "the 1868 general election in the borough of Blackburn"
        }},
        {{
            "name": "16 March 1869",
            "type": "Time",
            "desc": "On 16 March 1869"
        }},
        {{
            "name": "Void",
            "type": "Attribute",
            "desc": "null and void"
        }},
        {{
            "name": "Election petition",
            "type": "Event",
            "desc": "an election petition had been lodged"
        }},
        {{
            "name": "William Henry Hornby",
            "type": "Person",
            "desc": "William Henry Hornby"
        }},
        {{
            "name": "Joseph Feilden",
            "type": "Person",
            "desc": "Joseph Feilden"
        }},
        {{
            "name": "by-election of 1869",
            "type": "Event",
            "desc": "the by-election of 1869"
        }},
        {{
            "name": "by-election time",
            "type": "Time",
            "desc": "31 March 1869"
        }},
        {{
            "name": "Edward Hornby",
            "type": "Person",
            "desc": "Edward Hornby"
        }}
    ],
    "relations": [
        {{
            "src": "Henry Master Feilden",
            "dst": "Feilden's Birthday",
            "type": "Fact",
            "desc": "Feilden was born on 21 February 1818"
        }},
        {{
            "src": "Henry Master Feilden",
            "dst": "English",
            "type": "Fact",
            "desc": "Feilden is an English"
        }},
        {{
            "src": "Henry Master Feilden",
            "dst": "Conservative Party",
            "type": "Fact",
            "desc": "Feilden is a member of the Conservative Party"
        }},
        {{
            "src": "Henry Master Feilden",
            "dst": "Politician",
            "type": "Fact",
            "desc": "Feilden is a politician"
        }},
        {{
            "src": "1868 General Election",
            "dst": "16 March 1869",
            "type": "Time",
            "desc": "The 1868 general election on 16 March 1869"
        }},
        {{
            "src": "1868 General Election",
            "dst": "Void",
            "type": "Action",
            "desc": "The 1868 general election was declared null and void"
        }},
        {{
            "src": "Election petition",
            "dst": "Void",
            "type": "Step",
            "desc": "The election was declared null and void after an election petition had been lodged"
        }},
        {{
            "src": "Joseph Feilden",
            "dst": "Henry Master Feilden",
            "type": "Fact",
            "desc": "Joseph Feilden was the father of Henry Master Feilden"
        }},
        {{
            "src": "1868 General Election",
            "dst": "William Henry Hornby",
            "type": "Action",
            "desc": "William Henry Hornby was elected in the 1868 general election"
        }},
        {{
            "src":"1868 General Election",
            "dst":"Joseph Feilden",
            "type":"Action",
            "desc":"Joseph Feilden was elected in the 1868 general election"
        }},
        {{
            "src": "by-election of 1869",
            "dst": "by-election time",
            "type": "Time",
            "desc": "The by-election of 1869 was held on 31 March 1869"
        }},
        {{
            "src": "by-election of 1869",
            "dst": "Edward Hornby",
            "type": "Action",
            "desc": "Edward Hornby was elected in the by-election of 1869"
        }},
        {{
            "src": "by-election of 1869",
            "dst": "Henry Master Feilden",
            "type": "Action",
            "desc": "Joseph Feilden was elected in the by-election of 1869"
        }}
    ]
}}
"""
]

_temp_str = '\n'.join(_graph_building_example_list)

_graph_building_example_template = f"-Example-\n{_temp_str}"

_graph_building_query_template = """
-Query-
[Input]
title: {input_title}
text: {input_text}
prop: {input_prop}
[Output]
"""

graph_building = PromptTemplate(
    input_variables=["entity_types", "relation_types", "input_title", "input_text", "input_prop"],    
    # template=_graph_building_info_template + _graph_building_example_template + _graph_building_query_template,
    template=_graph_building_info_template + _graph_building_query_template,
)

graph_building_without_type = PromptTemplate(
    input_variables=["input_title", "input_text", "input_prop"],    
    # template=_graph_building_info_template + _graph_building_example_template + _graph_building_query_template,
    template=_graph_building_info_without_type_template + _graph_building_query_template,
)

simple_graph_building_template = """Your are a graph generator, you need to convert the text into a graph.
-Steps-
1. Identify the entities in the text. You need to analyze sentence by sentence to get as many entities in the text as possible.
You must extract the following information:
- Entity name: each entity should give an individual name.
- Ontology type of the entity: you must to classify the entity into one of the following categories: [{entity_types}].
- Entity description: Comprehensive description of the entity's attributes and activities.
2. Identify the relations between the entities. You need to analyze sentence by sentence to get as many relations in the text as possible.
You must extract the following information:
- Source entity: the source entity of the relation.
- Destination entity: the destination entity of the relation.
- Ontology type of the relation: you must to classify the relation into one of the following categories: [{relation_types}].
- Relation description: Comprehensive description of the relation's attributes and activities.


-Query-
[Input]
title: {input_title}
text: {input_text}
prop: {input_prop}
[Output]
"""

simple_graph_building = PromptTemplate(
    input_variables=["entity_types", "relation_types", "input_title", "input_text", "input_prop"],
    template=simple_graph_building_template,
)

graph_records_template = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity", <entity_name>, <entity_type>, <entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_type: One of the following types: [{relation_types}]
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship", <source_entity>, <target_entity>, <relation_type>, <relation_description>) that are *clearly related* to each other.

3. Identify all relationships among the input text and identified entities. Each attribute should be in the form of:
- attribute_key: the key of the attribute.
- attribute_value: the value of the attribute.
- attribute_description: Comprehensive description of the attribute's attributes and activities
- attribute_entity: the entity that the attribute belongs to.
Format each attribute as ("attribute", <attribute_key>, <attribute_value>, <attribute_description>, <attribute_entity>) that are *clearly related* to each other.

4. Return output in English as a single list of all the entities and relationships identified in steps 1, 2 and 3. Use **; ** as the list delimiter.

5. When finished, output <END_OUTPUT>

######################
-Examples-
######################
Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity", "Alex", "person", "Alex is a character who experiences frustration and is observant of the dynamics among other characters."); 
("entity", "Taylor", "person", "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."); 
("entity", "Jordan", "person", "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."); 
("entity", "Cruz", "person", "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."); 
("entity", "The Device", "technology", "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."); 
("relationship", "Taylor", "The Device", "Fact", "Taylor shows a moment of reverence towards the device, indicating its significance."); 
("relationship", "Jordan", "The Device", "Fact", "Jordan has a significant interaction with Taylor regarding the device."); 
("relationship", "Alex", "Taylor", "Observation", "Alex observes Taylor's change in perspective towards the device."); 
("relationship", "Alex", "Jordan", "Observation", "Alex observes Jordan's interaction with Taylor regarding the device."); 
("relationship", "Cruz", "Control", "Conceptual", "Cruz's vision of control and order influences the dynamics among the characters."); 
("relationship", "Taylor", "Jordan", "Interaction", "Taylor and Jordan share a moment of mutual understanding regarding the device."); 
("attribute", "authoritarian_certainty", "Taylor", "Taylor is portrayed with authoritarian certainty, influencing their interactions with others.", "Taylor"); 
("attribute", "commitment_to_discovery", "Jordan", "Jordan shares a commitment to discovery, which drives their actions.", "Jordan"); 
("attribute", "competitive_undercurrent", "Alex", "Alex observes the competitive undercurrent among the characters.", "Alex"); 
("attribute", "vision_of_control", "Cruz", "Cruz is associated with a vision of control and order.", "Cruz"); 
("attribute", "reverence_for_device", "Taylor", "Taylor shows a moment of reverence towards the device, indicating its importance.", "Taylor"); 
("attribute", "interaction_with_taylor", "Jordan", "Jordan has a significant interaction with Taylor regarding the device.", "Jordan"); 
<END_OUTPUT>
#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

# Example 2:

# Text:
# They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

# Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

# Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
# #############
# Output:
# ("entity", "Washington", "location", "Washington is a location where communications are being received, indicating its importance in the decision-making process."); 
# ("entity", "Operation: Dulce", "mission", "Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."); 
# ("entity", "The team", "organization", "The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."); 
# ("relationship", "The team", "Operation: Dulce", "Fact", "The team is actively involved in Operation: Dulce, which has evolved to interact and prepare."); 
# ("relationship", "Washington", "The team", "Communication", "The team receives communications from Washington, indicating its importance in their mission."); 
# ("attribute", "mission_evolution", "Operation: Dulce", "Operation: Dulce has evolved to interact and prepare, indicating a significant shift in objectives and activities.", "Operation: Dulce"); 
# ("attribute", "communication_location", "Washington", "Washington is a key location for communications in the mission.", "The team"); 
# ("attribute", "role_change", "The team", "The team has transitioned from passive observers to active participants in the mission.", "The team");
# <END_OUTPUT>
# #############################
# Example 3:

# Text:
# their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

# "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

# Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

# Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

# The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
# #############
# Output:
# ("entity", "Sam Rivera", "person", "Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."); 
# ("entity", "Alex", "person", "Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."); 
# ("entity", "Control", "concept", "Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."); 
# ("entity", "Intelligence", "concept", "Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."); 
# ("entity", "First Contact", "event", "First Contact is the potential initial communication between humanity and an unknown intelligence."); 
# ("entity", "Humanity's Response", "event", "Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."); 
# ("relationship", "Alex", "Team", "Observation", "Alex observes the dynamics and interactions within the team."); 
# ("relationship", "Sam Rivera", "Alex", "Colleague", "Sam Rivera and Alex are part of the same team working on first contact."); 
# ("relationship", "Intelligence", "Control", "Conceptual", "The Intelligence challenges the concept of Control by writing its own rules."); 
# ("attribute", "leadership", "Alex", "Alex is the leader of the team attempting first contact with the unknown intelligence.", "Alex"); 
# ("attribute", "emotion", "awe and anxiety", "Sam Rivera experiences a mix of awe and anxiety while working on communicating with the unknown intelligence.", "Sam Rivera"); 
# ("attribute", "role", "team member", "Sam Rivera is a team member involved in the first contact mission.", "Sam Rivera"); 
# ("attribute", "concept", "Control", "Control is a concept challenged by the unknown intelligence.", "Control"); 
# <END_OUTPUT>

graph_entity_records_template = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity", <entity_name>, <entity_type>, <entity_description>

2. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **; ** as the list delimiter.

3. When finished, output <END_OUTPUT>

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity", "Alex", "person", "Alex is a character who experiences frustration and is observant of the dynamics among other characters."); 
("entity", "Taylor", "person", "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."); 
("entity", "Jordan", "person", "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."); 
("entity", "Cruz", "person", "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."); 
("entity", "The Device", "technology", "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."); 
<END_OUTPUT>
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity", "Washington", "location", "Washington is a location where communications are being received, indicating its importance in the decision-making process."); 
("entity", "Operation: Dulce", "mission", "Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."); 
("entity", "The team", "organization", "The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."); 
<END_OUTPUT>
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity", "Sam Rivera", "person", "Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."); 
("entity", "Alex", "person", "Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."); 
("entity", "Control", "concept", "Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."); 
("entity", "Intelligence", "concept", "Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."); 
("entity", "First Contact", "event", "First Contact is the potential initial communication between humanity and an unknown intelligence."); 
("entity", "Humanity's Response", "event", "Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."); 
<END_OUTPUT>
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""
graph_entity_records = PromptTemplate(
    input_variables=["entity_types", "input_text"],
    template=graph_entity_records_template,
)
graph_entity_records_msg = HumanMessagePromptTemplate(
    prompt=graph_entity_records,
    additional_kwargs={"input_variables": ["entity_types", "input_text"]},
)


graph_relation_records_template = """-Goal- 
Identify all relationships among the input text and identified entities.
-Steps-
1. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_type: One of the following types: [{relation_types}]
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship", <source_entity>, <target_entity>, <relation_type>, <relation_description>) that are *clearly related* to each other.
2. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **; ** as the list delimiter.
3. When finished, output <END_OUTPUT>

-Example-
("relationship", "Sam Rivera", "Intelligence", "Fact", "Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."); 
("relationship", "Alex", "First Contact", "Fact", "Alex leads the team that might be making the First Contact with the unknown intelligence."); 
("relationship", "Alex", "Humanity's Response",  "Fact", "Alex and his team are the key figures in Humanity's Response to the unknown intelligence."); 
("relationship", "Control", "Intelligence",  "Fact", "The concept of Control is challenged by the Intelligence that writes its own rules."); 

Input:
text: 
{input_text}

entities: 
{entities}

Output:
"""

graph_relation_records = PromptTemplate(
    input_variables=["entity_types"],
    template=graph_relation_records_template,
)
graph_relation_records_msg = HumanMessagePromptTemplate(
    prompt=graph_relation_records,
    additional_kwargs={"input_variables": ["relation_types"]},
)

graph_attributes_records_template = """-Goal- 
Identify all attributes of the identified entities.
-Steps-
1. Find all attributes of the identified entities. Each attribute should be in the form of:
- attribute_key: the key of the attribute.
- attribute_value: the value of the attribute.
- attribute_description: Comprehensive description of the attribute's attributes and activities
- attribute_entity: the entity that the attribute belongs to.
Format each attribute as ("attribute", <attribute_key>, <attribute_value>, <attribute_description>, <attribute_entity>) that are *clearly related* to each other.
2. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **; ** as the list delimiter.
3. When finished, output <END_OUTPUT>
######################

-Example-
("attribute", "birth_date", "21 February 1818", "The birth date of Henry Master Feilden", "Henry Master Feilden"); 
("attribute", "death_date", "5 September 1875", "The death date of Henry Master Feilden", "Henry Master Feilden"); 
("attribute", "nationality", "English", "The nationality of Henry Master Feilden", "Henry Master Feilden"); 
("attribute", "party", "Conservative Party", "The political party affiliation of Henry Master Feilden", "Henry Master Feilden"); 
("attribute", "election_date", "31 March 1869", "The date of the by-election in which Henry Master Feilden was elected", "by-election of 1869"); 
("attribute", "relation_type", "Fact", "The type of relation between Henry Master Feilden and his father Joseph Feilden", "Joseph Feilden"); 

Input:
text: 
{input_text}

entities: 
{entities}

Output:
"""
graph_attributes_records = PromptTemplate(
    input_variables=[],
    template=graph_attributes_records_template,
)
graph_attributes_records_msg = HumanMessagePromptTemplate(
    prompt=graph_attributes_records,
    additional_kwargs={"input_variables": ["entity_types", "input_text"]},
)


graph_records = PromptTemplate(
    input_variables=["entity_types", "relation_types", "input_text"],
    template=graph_records_template,
)
