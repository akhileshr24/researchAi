
from src.rag.vectorstore import get_all_chunks

from src.rag.tests.evaluation_dataset import evaluation_dataset


def test_evaluation_dataset():

    chunks = get_all_chunks()

    available_chunks = {
        (chunk["page"], chunk["chunk_index"])
        for chunk in chunks
    }

    errors = []

    for case in evaluation_dataset:

        question = case["question"]
        answerable = case["answerable"]
        grades = case["relevance_grades"]

        if answerable and not grades:
            errors.append(
                f"{question}: missing relevance labels"
            )

        if not answerable and grades:
            errors.append(
                f"{question}: unexpected relevance labels"
            )

        for chunk_id, grade in grades.items():

            if chunk_id not in available_chunks:
                errors.append(
                    f"{question}: chunk {chunk_id} does not exist"
                )

            if grade not in (1, 2):
                errors.append(
                    f"{question}: invalid relevance grade {grade}"
                )

    if errors:
        for error in errors:
            print("ERROR:", error)

        raise AssertionError(
            "Evaluation dataset validation failed"
        )

    answerable_count = sum(
        case["answerable"]
        for case in evaluation_dataset
    )

    unanswerable_count = len(evaluation_dataset) - answerable_count

    print("Evaluation dataset validation passed!")
    print("Total questions:", len(evaluation_dataset))
    print("Answerable:", answerable_count)
    print("Unanswerable:", unanswerable_count)


if __name__ == "__main__":
    test_evaluation_dataset()
