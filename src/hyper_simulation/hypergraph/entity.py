from enum import Enum, auto
from hyper_simulation.hypergraph.linguistic import Entity

# PERSON: Human being, individual, or specific character.
# COUNTRY: A nation with its own government.
# LOC: Geographical location, natural region, body of water.
# ORG: Organization, institution, company, government body.
# FAC: Physical building, facility, structure.
# ​GPE​​: Geopolitical entity, such as cities, states, provinces (but not countries).
# NORP: Nationalities, religious or political groups.
# PRODUCT: Physical object, vehicle, device, manufactured good.
# WORK_OF_ART: Piece of art, publication, show.
# LAW: Legal document, binding agreement.
# LANGUAGE: Spoken or written human language.
# OCCUPATION: Job, profession, trade.
# EVENT: Phenomenon, historical event, sports match.
# TEMPORAL: Time period, specific date, unit of time.
# NUMBER: Mathematical number, quantity.
# CONCEPT: Abstract idea, theoretical concept.
# NOT_ENT: Use this if the synset does not fit any of the above 16 categories.
# ORGANISM: Living being, such as animal, plant, or microorganism.
# FOOD: Edible substance, dish, or cuisine.
# MEDICAL: Medical condition, disease, symptom, or treatment.
# ANATOMY: Body part, organ, or anatomical structure.
# SUBSTANCE: Chemical element, compound, or material.
# ASTRO: Astronomical object, such as a star, planet, or galaxy.
# AWARD: Prize, honor, or recognition given to a person or organization.
# VEHICLE: Means of transportation, such as a car, airplane, or bicycle.
# THEORY: Scientific or philosophical theory, principle, or framework.
# GROUP: Collection of individuals likes a family, team, class, or social group.
# FEATURE: Distinctive attribute, property, or characteristic of an entity or concept.
# ECONOMIC: Economic entity, such as a market, industry, or economic concept.
# SOCIOLOGY: Concepts related to society, culture, sociology, or social interactions, such as social media, social movement, or social issue.
# PHENOMENON: Natural or social phenomenon, such as climate change, pandemic, or cultural trend.
# ACTION: Action, behavior, or process, such as a specific activity, event, or process that is not covered by the above categories.
class ENT(Enum):
    PERSON = auto()
    COUNTRY = auto()
    LOC = auto()
    ORG = auto()
    FAC = auto()
    GPE = auto()
    NORP = auto()
    PRODUCT = auto()
    WORK_OF_ART = auto()
    LAW = auto()
    LANGUAGE = auto()
    OCCUPATION = auto()
    EVENT = auto()
    TEMPORAL = auto()
    NUMBER = auto()
    CONCEPT = auto()
    ORGANISM = auto()
    FOOD = auto()
    MEDICAL = auto()
    ANATOMY = auto()
    SUBSTANCE = auto()
    ASTRO = auto()
    AWARD = auto()
    VEHICLE = auto()
    THEORY = auto()
    GROUP = auto()
    FEATURE = auto()
    ECONOMIC = auto()
    SOCIOLOGY = auto()
    PHENOMENON = auto()
    ACTION = auto()
    NOT_ENT = auto()

    @staticmethod
    def from_entity(ent: Entity) -> "ENT":
        mapping = {
            Entity.PERSON: ENT.PERSON,
            Entity.NORP: ENT.NORP,
            Entity.FAC: ENT.FAC,
            Entity.ORG: ENT.ORG,
            Entity.GPE: ENT.GPE,
            Entity.LOC: ENT.LOC,
            Entity.PRODUCT: ENT.PRODUCT,
            Entity.EVENT: ENT.EVENT,
            Entity.WORK_OF_ART: ENT.WORK_OF_ART,
            Entity.LAW: ENT.LAW,
            Entity.LANGUAGE: ENT.LANGUAGE,

            # spaCy's temporal entities are mapped to the unified temporal bucket.
            Entity.DATE: ENT.TEMPORAL,
            Entity.TIME: ENT.TEMPORAL,

            # Numeric-like entities are mapped to the unified number bucket.
            Entity.PERCENT: ENT.NUMBER,
            Entity.MONEY: ENT.NUMBER,
            Entity.QUANTITY: ENT.NUMBER,
            Entity.ORDINAL: ENT.NUMBER,
            Entity.CARDINAL: ENT.NUMBER,

            Entity.NOT_ENTITY: ENT.NOT_ENT,
        }
        return mapping.get(ent, ENT.NOT_ENT)

    def level(self) -> int:
        """
        Define a hierarchy level for each entity type to assist in type inference during fusion.
        Lower number means more general, higher number means more specific.
        """
        hierarchy = {
            ENT.NOT_ENT: 0,
            ENT.CONCEPT: 1,
            ENT.TEMPORAL: 4,
            ENT.NUMBER: 2,
            ENT.ORGANISM: 3,
            ENT.FOOD: 3,
            ENT.MEDICAL: 3,
            ENT.ANATOMY: 3,
            ENT.SUBSTANCE: 3,
            ENT.ASTRO: 3,
            ENT.AWARD: 3,
            ENT.VEHICLE: 3,
            ENT.PERSON: 4,
            ENT.COUNTRY: 5,
            ENT.LOC: 4,
            ENT.ORG: 4,
            ENT.FAC: 4,
            ENT.GPE: 4,
            ENT.NORP: 4,
            ENT.PRODUCT: 2,
            ENT.WORK_OF_ART: 3,
            ENT.LAW: 4,
            ENT.LANGUAGE: 4,
            ENT.OCCUPATION: 4,
            ENT.EVENT: 2,
            ENT.THEORY: 3,
            ENT.GROUP: 4,
            ENT.FEATURE: 3,
            ENT.ECONOMIC: 3,
            ENT.SOCIOLOGY: 3,
            ENT.PHENOMENON: 3,
            ENT.ACTION: 2
        }
        return hierarchy.get(self, 0)
    
    @staticmethod
    def from_str(label: str) -> "ENT":
        try:
            return ENT[label]
        except KeyError:
            return ENT.NOT_ENT
    
    