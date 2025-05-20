from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_qa_lengths(file_path: Path) -> List[Tuple[int, int]]:
    lengths = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        print(f"Warning: Line {line_num} is not a JSON object")
                        continue

                    prompt = item.get("prompt")
                    response = item.get("response")

                    if prompt is None or response is None:
                        print(f"Warning: Missing prompt or response at line {line_num}")
                        continue

                    lengths.append((len(str(prompt).strip()), len(str(response).strip())))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num}: {e}")

        return lengths
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []


def plot_length_distribution(lengths: List[Tuple[int, int]]):
    if not lengths:
        print("No data to plot")
        return

    prompts, responses = zip(*lengths)

    plt.figure(figsize=(10, 6))
    plt.hist(responses, bins=50, alpha=0.7, label='Response length')
    plt.hist(prompts, bins=50, alpha=0.7, label='Prompt length')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title("Prompt/Response Length Distribution")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "data" / "healthcare_qa.jsonl"
    lengths = load_qa_lengths(file_path)
    plot_length_distribution(lengths)