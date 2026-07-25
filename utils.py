import re
def extract_final_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\-0-9,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-0-9,\.]+", text)
    return numbers[-1].replace(",", "").strip() if numbers else None