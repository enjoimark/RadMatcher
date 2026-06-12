"""Train the matcher and save it."""

from matcher import SimpleMatcher

if __name__ == "__main__":
    print("=" * 80)
    print("Training SimpleMatcher")
    print("=" * 80)
    print()

    matcher = SimpleMatcher.build(
        codes_csv="exam_codes.csv",
        mappings_csv="exam_mappings.csv",
        train_ml=True
    )

    print()
    print("=" * 80)
    print("Saving model")
    print("=" * 80)
    matcher.save("matcher_model.pkl")

    print()
    print("=" * 80)
    print("Testing")
    print("=" * 80)
    print()

    # Test variations
    test_queries = [
        "cr chest 2v",
        "CR 2 VIEW CHEST",
        "XR CHEST PA AND LATERAL",
        "chest 2 view pa lat",
        "US OB 2nd trimester",
        "US OB 3rd trimester transvaginal",
        "US OB over 14 weeks twins",
    ]

    for query in test_queries:
        print(f'\nQuery: "{query}"')
        results = matcher.match(query, max_results=3)
        for i, r in enumerate(results, 1):
            print(f'  {i}. [{r["score"]:3d}] {r["code"]}: {r["description"]} ({r.get("views", "?")} views)')

    print()
    print("DONE! Model saved to matcher_model.pkl")
