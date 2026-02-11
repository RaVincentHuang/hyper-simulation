import re

def clean_text_for_spacy(text: str) -> str:
    if not text:
        return ""
    
    # 1. 将所有类型的空白字符（包括转义字符、全角空格、不换行空格）替换为标准半角空格
    # \s 在正则中匹配 [ \t\n\r\f\v] 以及 Unicode 定义的所有空格
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 去除首尾空格
    text = text.strip()
    
    return text

if __name__ == "__main__":
    raw_text = """Dekhmeh Rawansar.
Dekhmeye Rawansar (دخمه روانسر) is a rock-cut tomb located near the town of Ravansar (Kurdish: Rowansar), about 57 km northwest of Kermanshah, at west of Iran.
This tomb was known to Ernst Herzfeld but he never visited it.
The first archaeologist who visited the tomb was Massoud Golzari, an Iranian archaeologist who attributed it to Medes.
It is re-visited and examined by Peter Calmeyer, German archaeologist (birth.
5 September 1930 in Halle, death.
22 November 1995 in Berlin) in the 1970s, who according to his observations related the tomb to the Achaemenid period.
"""
    cleaned_text = clean_text_for_spacy(raw_text)
    print(f"'{cleaned_text}'") 
