import json
from pathlib import Path
from experiment import replay

if __name__ == "__main__":
    sample = json.loads(Path(__file__).with_name("samples.json").read_text())
    result = replay(sample["tape"], sample["horizon"])
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
