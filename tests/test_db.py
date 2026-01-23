import conceptnet_lite
from conceptnet_lite import Label
import peewee

def diagnose_apple(db_path="conceptnet.db"):
    print(f"🔍 Diagnosing 'Apple' in {db_path}...")
    try:
        conceptnet_lite.connect(db_path)
        
        # 1. 查 Label
        term = "apple"
        try:
            label = Label.get(text=term, language='en')
            print(f"✅ Label found: {label.text}")
        except peewee.DoesNotExist:
            print("❌ Label 'apple' NOT found. (Check DB integrity)")
            return

        # 2. 遍历所有边，专门看 IsA
        print("\n--- Scanning Outgoing Edges ---")
        found_isa = False
        
        for concept in label.concepts:
            for edge in concept.edges_out:
                # 获取原始属性
                rel_name = edge.relation.name
                rel_uri = edge.relation.uri
                target = edge.end.text
                
                # 打印所有看似 IsA 的关系
                # 检查 name 是否包含 'is', 'a' 或者 uri 包含 'IsA'
                if 'isa' in rel_name.lower().replace("_", "") or 'isa' in rel_uri.lower():
                    found_isa = True
                    print(f"🎯 HIT: Target='{target}'")
                    print(f"       .name property: '{rel_name}'  <-- 关键看这里")
                    print(f"       .uri  property: '{rel_uri}'")
                    print(f"       Target Language: {edge.end.language.name}")
                    print("-" * 30)
                    
        if not found_isa:
            print("⚠️ No IsA edges found using loose matching.")
            print("List of ALL relations found (to see what went wrong):")
            for concept in label.concepts:
                for edge in concept.edges_out:
                    print(f" - {edge.relation.name} ({edge.relation.uri})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_apple()