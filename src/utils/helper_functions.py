def clean_text(text):
    """简单文本清洗：去除特殊符号、统一大小写"""
    import re
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

def map_labels(label):
    """将标签映射为整数（0/1）"""
    return 1 if label == "True" else 0
