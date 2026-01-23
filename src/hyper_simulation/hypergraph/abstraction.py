import spacy
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk
from spacy.tokens import Token


ANCHORS = [
    # --------------------------------------------------------------------------
    # 1. TECHNOLOGY & DIGITAL (技术与数字资产)
    # --------------------------------------------------------------------------
    ("AI_Model",            {"artificial_intelligence.n.01", "expert_system.n.01", "neural_network.n.01"}),
    ("Software",            {"software.n.01", "computer_program.n.01", "app.n.01", "operating_system.n.01", "browser.n.01"}),
    ("Database",            {"database.n.01", "information_system.n.01"}),
    ("CyberThreat",         {"computer_virus.n.01", "malware.n.01", "cyberterrorism.n.01", "security_hole.n.01"}),
    ("Network",             {"computer_network.n.01", "internet.n.01", "web.n.01", "intranet.n.01", "server.n.03"}),
    ("MobileDevice",        {"cellular_telephone.n.01", "tablet.n.03"}), 
    ("ComputerHardware",    {"computer.n.01", "peripheral.n.01", "processor.n.01", "chip.n.01", "screen.n.01"}),
    ("DigitalData",         {"data.n.01", "file.n.01", "dataset.n.01", "digital_image.n.01", "code.n.03"}),

    # --------------------------------------------------------------------------
    # 2. FINANCE & ECONOMY (金融与经济)
    # --------------------------------------------------------------------------
    ("Currency",            {"currency.n.01", "money.n.01", "cryptocurrency.n.01"}), # 注意: Crypto在老WN可能没有，需依赖NER
    ("FinancialInst",       {"financial_institution.n.01", "bank.n.02", "exchange.n.06"}), # 银行/交易所
    ("FinancialAsset",      {"stock.n.01", "bond.n.01", "security.n.04", "asset.n.01", "fund.n.01"}),
    ("Tax",                 {"tax.n.01", "tariff.n.01", "levy.n.01"}),
    ("EconomicMetric",      {"gross_domestic_product.n.01", "inflation.n.01", "interest_rate.n.01", "budget.n.01"}),
    ("Transaction",         {"transaction.n.01", "payment.n.01", "investment.n.01", "loan.n.01"}),

    # --------------------------------------------------------------------------
    # 3. HEALTHCARE & LIFE SCIENCES (医疗与生命科学)
    # --------------------------------------------------------------------------
    ("Disease",             {"disease.n.01", "illness.n.01", "syndrome.n.01", "infection.n.01"}),
    ("Medication",          {"drug.n.01", "medicine.n.02", "vaccine.n.01", "antibiotic.n.01"}),
    ("MedicalDevice",       {"medical_instrument.n.01", "implant.n.01", "diagnostic.n.01"}),
    ("BodyPart",            {"body_part.n.01", "organ.n.01"}),
    ("Microorganism",       {"microorganism.n.01", "virus.n.01", "bacterium.n.01"}),
    ("HealthcareFacility",  {"hospital.n.01", "clinic.n.01", "pharmacy.n.01"}),

    # --------------------------------------------------------------------------
    # 4. MANUFACTURING & MATERIALS (制造与材料)
    # --------------------------------------------------------------------------
    ("Vehicle_Land",        {"car.n.01", "truck.n.01", "bus.n.01", "train.n.01", "locomotive.n.01"}),
    ("Vehicle_Air",         {"aircraft.n.01", "plane.n.01", "drone.n.01", "helicopter.n.01"}),
    ("Vehicle_Water",       {"vessel.n.03", "ship.n.01", "boat.n.01", "submarine.n.01"}),
    ("Machine",             {"machine.n.01", "engine.n.01", "motor.n.01", "robot.n.01", "generator.n.01"}),
    ("Tool",                {"tool.n.01", "implement.n.01", "device.n.01"}),
    ("RawMaterial",         {"raw_material.n.01", "ore.n.01", "timber.n.01"}),
    ("Metal",               {"metal.n.01", "alloy.n.01", "steel.n.01", "gold.n.03"}),
    ("Chemical",            {"chemical.n.01", "compound.n.02", "element.n.01", "acid.n.01"}),
    ("Textile",             {"fabric.n.01", "cloth.n.01"}),
    ("EnergySource",        {"fossil_fuel.n.01", "electricity.n.01", "solar_energy.n.01", "oil.n.01", "gas.n.02"}),

    # --------------------------------------------------------------------------
    # 5. LEGAL, GOV & SECURITY (法律、政府与安全)
    # --------------------------------------------------------------------------
    ("LegalDocument",       {"contract.n.01", "treaty.n.01", "will.n.03", "license.n.01", "constitution.n.01"}),
    ("LawRegulation",       {"law.n.02", "regulation.n.01", "statute.n.01", "act.n.02"}),
    ("GovernmentBody",      {"government.n.01", "ministry.n.01", "agency.n.01", "court.n.01", "council.n.01"}),
    ("MilitaryUnit",        {"military_unit.n.01", "army.n.01", "navy.n.01"}),
    ("Weapon",              {"weapon.n.01", "gun.n.01", "missile.n.01", "bomb.n.01"}),
    ("Crime",               {"crime.n.01", "felony.n.01", "fraud.n.01", "theft.n.01"}),

    # --------------------------------------------------------------------------
    # 6. ROLES & PEOPLE (角色与人物)
    # --------------------------------------------------------------------------
    ("Executive",           {"executive.n.01", "director.n.01", "chief.n.01", "president.n.04"}), # CEO/CFO级别
    ("Official",            {"official.n.01", "politician.n.01", "diplomat.n.01", "judge.n.01"}),
    ("Professional",        {"professional.n.01", "expert.n.01", "engineer.n.01", "lawyer.n.01", "scientist.n.01", "doctor.n.01"}),
    ("Worker",              {"worker.n.01", "laborer.n.01", "employee.n.01"}),
    ("Consumer",            {"consumer.n.01", "customer.n.01", "user.n.01", "client.n.01"}),
    ("Family",              {"relative.n.01", "parent.n.01", "child.n.01", "spouse.n.01"}),

    # --------------------------------------------------------------------------
    # 7. LOCATION & INFRASTRUCTURE (地点与基建)
    # --------------------------------------------------------------------------
    ("GeoPoliticalEntity",  {"country.n.02", "state.n.01", "province.n.01", "nation.n.02"}),
    ("CityTown",            {"city.n.01", "town.n.01", "village.n.01", "capital.n.03"}),
    ("Infrastructure",      {"infrastructure.n.02", "road.n.01", "bridge.n.01", "tunnel.n.01", "railway.n.01"}),
    ("TransitHub",          {"airport.n.01", "station.n.01", "port.n.01", "harbor.n.01"}),
    ("Building_Commercial", {"office_building.n.01", "skyscraper.n.01", "store.n.01", "hotel.n.01"}),
    ("Building_Industrial", {"factory.n.01", "plant.n.01", "warehouse.n.01", "power_station.n.01"}),
    ("Building_Residential",{"house.n.01", "apartment.n.01"}),
    ("NaturalFeature",      {"mountain.n.01", "river.n.01", "ocean.n.01", "forest.n.01", "lake.n.01"}),

    # --------------------------------------------------------------------------
    # 8. ABSTRACT & METRICS (抽象与度量)
    # --------------------------------------------------------------------------
    ("TimeDuration",        {"time_period.n.01", "year.n.01", "month.n.01", "century.n.01"}),
    ("TimePoint",           {"date.n.01", "day.n.01", "moment.n.01"}),
    ("QuantityMetric",      {"measure.n.02", "quantity.n.01", "percentage.n.01", "amount.n.03", "distance.n.01"}),
    ("Methodology",         {"method.n.01", "technique.n.01", "procedure.n.01", "algorithm.n.01", "strategy.n.01"}),
    ("IntellectualProp",    {"copyright.n.01", "patent.n.01", "trademark.n.01"}),
    ("Event_Disaster",      {"catastrophe.n.01", "disaster.n.01", "earthquake.n.01", "flood.n.01", "fire.n.01"}),
    ("Event_Social",        {"social_event.n.01", "ceremony.n.01", "conference.n.01", "meeting.n.01"}),
    ("Language",            {"language.n.01"}),

    # --------------------------------------------------------------------------
    # 9. GENERAL FALLBACKS (通用兜底)
    # --------------------------------------------------------------------------
    ("Company",             {"company.n.01", "enterprise.n.02", "firm.n.01"}),
    ("Organization",        {"organization.n.01", "group.n.01"}),
    ("Person",              {"person.n.01"}),
    ("Location",            {"location.n.01", "place.n.01"}),
    ("Product",             {"product.n.02", "commodity.n.01", "merchandise.n.01"}),
    ("Artifact",            {"artifact.n.01", "physical_entity.n.01"}),
    ("Concept",             {"abstraction.n.06", "idea.n.01"})
]

COMPATIBILITY_MATRIX = {
    # --- 人物与社会群体 ---
    "PERSON": {
        "Professional", "Worker", "Leader", "Customer", "Politician", 
        "Artist", "Athlete", "Scientist", "Criminal"
    },
    "NORP": { # Nationalities, Religious, Political groups
        "SocialGroup", "PoliticalParty", "Religion", "Ethnicity", "Team", 
        "Organization" # 有时 NORP 也可以被视为一种松散组织
    },

    # --- 机构与设施 ---
    "ORG": {
        "Company", "Government", "Team", "NonProfit", "FinancialInst", 
        "University", "Media", "PoliticalParty", "School", "Agency"
    },
    "FAC": { # Buildings, Airports, Highways
        "Building", "Infrastructure", "Airport", "Station", "Road", 
        "Bridge", "Factory", "Hospital", "Structure"
    },

    # --- 地理与行政 ---
    "GPE": { # Countries, Cities, States
        "City", "Country", "Province", "State", "Municipality", "Nation"
    },
    "LOC": { # Non-GPE locations (Mountains, Bodies of Water)
        "NaturalPlace", "Mountain", "BodyOfWater", "Continent", "Region", "Land"
    },

    # --- 人造物与产品 ---
    "PRODUCT": {
        "Vehicle", "Software", "Hardware", "Machine", "Weapon", 
        "Food", "Chemical", "Drug", "Clothing", "Furniture", "Tool", 
        "Aircraft", "Watercraft"
    },
    "WORK_OF_ART": { # Books, Songs, Movies
        "Document", "Book", "Movie", "Song", "Painting", "Sculpture", 
        "Game", "Software", "Show", "Publication"
    },

    # --- 法律与语言 ---
    "LAW": {
        "Regulation", "Constitution", "Treaty", "Act", "Rule"
    },
    "LANGUAGE": {
        "Language", "Communication"
    },

    # --- 时间 ---
    "DATE": {
        "TimePeriod", "Year", "Month", "Day", "Date"
    },
    "TIME": {
        "TimePeriod", "Time", "Duration", "Hour"
    },

    # --- 事件 ---
    "EVENT": {
        "Incident", "Disaster", "SocialEvent", "SportsEvent", "War", 
        "Competition", "Festival", "Ceremony"
    },

    # --- 数值与度量 (通常 WSD 对这些词效果不明显，主要靠 Regex 或 NER 本身) ---
    "PERCENT": {"Metric", "Percentage", "Rate"},
    "MONEY":   {"Currency", "Money", "Price", "Cost"},
    "QUANTITY": {"Metric", "Distance", "Weight", "Volume", "Area", "Speed"},
    
    # --- 序数与基数 (通常不作为实体，但如果需要抽象) ---
    "ORDINAL": {"Order", "Rank"},
    "CARDINAL": {"Number", "Count", "Amount"}
}

DEFAULT_MAPPING = {
    "PERSON":       "Person",
    "NORP":         "Group",          # 民族/宗教归为群体
    "FAC":          "Structure",      # 设施归为结构
    "ORG":          "Organization",
    "GPE":          "GeoPolitical",
    "LOC":          "Location",
    "PRODUCT":      "Product",
    "EVENT":        "Event",
    "WORK_OF_ART":  "CreativeWork",   # 艺术作品
    "LAW":          "Regulation",
    "LANGUAGE":     "Language",
    "DATE":         "Time",
    "TIME":         "Time",
    "PERCENT":      "Metric",
    "MONEY":        "Currency",
    "QUANTITY":     "Metric",
    "ORDINAL":      "Value",
    "CARDINAL":     "Value"
}

class TokenAbstractor:
    def __init__(self):
        self.anchors = ANCHORS
        # 为了加速匹配，可以将 anchor set 预处理，但这里为了逻辑清晰保持原样

    def _spacy_to_wn_pos(self, spacy_pos):
        """将 SpaCy POS 转换为 WordNet POS"""
        if spacy_pos in ['NOUN', 'PROPN']: return wn.NOUN
        if spacy_pos == 'VERB': return wn.VERB
        if spacy_pos == 'ADJ': return wn.ADJ
        if spacy_pos == 'ADV': return wn.ADV
        return None
    
    def _get_contextual_synset(self, token: Token, doc):
        """
        核心逻辑：使用 PyWSD 进行词义消歧
        """
        wn_pos = self._spacy_to_wn_pos(token.pos_)
        if not wn_pos:
            return None

        # A. 尝试使用 Lesk 算法结合上下文 (Doc文本)
        try:
            # simple_lesk(句子文本, 目标词, 词性)
            synset = simple_lesk(doc.text, token.text, pos=wn_pos)
        except Exception:
            synset = None

        # B. 兜底机制：如果 Lesk 失败 (比如句子太短无重叠词)，回退到 MFS (最常用义项)
        if not synset:
            synsets = wn.synsets(token.lemma_, pos=wn_pos)
            if synsets:
                synset = synsets[0]
        
        return synset

    def get_abstraction(self, token: Token, doc) -> str:
        """
        输入 Token 和上下文 Doc，返回类型
        """
        # 1. 代词处理
        if token.pos_ == "PRON":
            if token.lower_ in ["he", "she", "who", "i"]: return "Person"
            return "Object"

        # 2. 获取经过消歧的 Synset
        # target_synset = None
        # if token.pos_ in ["NOUN", "PROPN"]:
        target_synset = self._get_contextual_synset(token, doc)
        
        wn_cat = None
        if target_synset:
            # 获取路径并匹配锚点
            hypernyms = set()
            for path in target_synset.hypernym_paths():
                for node in path:
                    hypernyms.add(node.name())
            
            # 匹配锚点
            for category, anchor_set in self.anchors:
                if not hypernyms.isdisjoint(anchor_set):
                    wn_cat = category
                    break # 找到最高优先级的分类即停止

        ner_tag = token.ent_type_

        # 3. 混合决策 (NER + WSD)
        if ner_tag:
            allowed_set = COMPATIBILITY_MATRIX.get(ner_tag, None)
            if wn_cat and allowed_set and wn_cat in allowed_set:
                return wn_cat
            
            return DEFAULT_MAPPING.get(ner_tag, wn_cat if wn_cat else "Object")

        # 4. 纯 WordNet 结果
        if wn_cat:
            return wn_cat
            
        # 5. 最终兜底
        return "Object" if token.pos_ in ["NOUN", "PROPN"] else "Unknown"
    
    def get_abstraction_path(self, token: Token, doc) -> list[str]:
        """
        获取纯 WordNet 的上位词路径 (支持 名词、动词、形容词、副词)。
        
        Returns:
            list[str]: 
            - 名词/动词: ['entity.n.01', ..., 'dog.n.01']
            - 形容词/副词: ['fast.a.01'] (通常只有一层)
        """
        # 1. 映射 POS
        wn_pos = self._spacy_to_wn_pos(token.pos_)
        if not wn_pos:
            # 如果是连词、介词等 WordNet 不收录的词，返回空或特定标识
            return [] 

        # 2. 获取上下文相关的 Synset
        # PyWSD 支持所有这些词性的消歧
        synset = self._get_contextual_synset(token, doc)
        
        # 3. 如果无法识别 (WordNet收录不全)，返回空路径或自身
        if not synset:
            # 这里的策略是：如果找不到 Synset，就无法建立路径
            # 你也可以选择返回 ['entity.n.01'] 作为绝对兜底，但对于动词/形容词这可能不准确
            return []

        # 4. 获取上位词路径
        # 注意: Adjective 和 Adverb 在 WordNet 中通常没有 hypernym_paths
        paths = synset.hypernym_paths()
        
        if not paths:
            # Case A: 形容词/副词，或者位于根节点的词
            # 直接返回它自己，作为路径的唯一节点
            return [synset.name()]

        # Case B: 名词/动词，通常有路径
        # 取第一条路径
        raw_path = paths[0]
        
        # 5. 转换为字符串列表
        return [node.name() for node in raw_path]

# ==============================================================================
# 4. 测试与演示
# ==============================================================================
if __name__ == "__main__":
    abstractor = TokenAbstractor()
    nlp = spacy.load('en_core_web_trf')
    # 定义不同工业领域的测试句
    test_cases = [
        ("IT/Tech", "The Python script crashed the server due to a memory leak."),
        ("Business", "The CEO of Google signed the new contract yesterday."),
        ("Logistics", "The freight truck delivered the containers to the warehouse."),
        ("Manufacturing", "The engineer checked the viscosity of the oil in the engine."),
        ("Finance", "The transaction fee was 5 dollars per user.")
    ]

    print(f"{'Token':<15} | {'Lemma':<12} | {'NER':<8} | {'Abstract Type (Result)'}")
    print("-" * 65)

    for domain, text in test_cases:
        print(f"--- Domain: {domain} ---")
        doc = nlp(text)
        
        # 仅展示名词性成分的抽象结果
        processed_tokens = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                abstract_type = abstractor.get_abstraction(token, doc)
                print(f"{token.text:<15} | {token.lemma_:<12} | {token.ent_type_:<8} | {abstract_type}")
        print("")
