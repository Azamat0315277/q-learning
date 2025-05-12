def calculate_average(scores: list[int]) -> float:
    return sum(scores) / len(scores)


def process_author_scores(author_name: str, scores: list[int]) -> float:
    average_score = calculate_average(scores)
    print(f"Author: {author_name}, Average Score: {average_score}")
    return average_score

def main() -> None:
    authors_scores = {
        "John": [10, 20, 30, 40, 50],
        "Jane": [10, 20, 30, 40, 50],
        "Jim": [10, 20, 30, 40, 50],
        "Jill": [10, 20, 30, 40, 50],
    }
    for author, scores in authors_scores.items():
        process_author_scores(author, scores)
        

if __name__ == "__main__":
    main()


