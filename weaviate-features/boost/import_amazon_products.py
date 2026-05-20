"""Import Amazon Products 2023 dataset into a local Weaviate instance.

Loads ~117,243 products from huggingface.co/datasets/milistu/AMAZON-Products-2023
with pre-computed 1536-dim embeddings, title, description, price, rating, and category.

Usage:
    uv run import_amazon_products.py [--limit N]
"""

import argparse

import weaviate
import weaviate.classes.config as wc
from datasets import load_dataset
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

COLLECTION_NAME = "AmazonProduct"


def create_collection(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vector_config=wc.Configure.Vectors.self_provided(),
        properties=[
            wc.Property(name="title", data_type=wc.DataType.TEXT),
            wc.Property(name="description", data_type=wc.DataType.TEXT),
            wc.Property(name="main_category", data_type=wc.DataType.TEXT),
            wc.Property(name="price", data_type=wc.DataType.NUMBER),
            wc.Property(name="average_rating", data_type=wc.DataType.NUMBER),
            wc.Property(name="rating_number", data_type=wc.DataType.NUMBER),
            wc.Property(name="date_first_available", data_type=wc.DataType.DATE),
            wc.Property(name="image", data_type=wc.DataType.TEXT),
        ],
    )
    print(f"Created collection '{COLLECTION_NAME}'")


def import_data(client: weaviate.WeaviateClient, limit: int) -> None:
    print("Downloading dataset (cached after first run)...")
    ds = load_dataset("milistu/AMAZON-Products-2023", split="train")

    collection = client.collections.get(COLLECTION_NAME)
    imported = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[cyan]{task.fields[imported]}[/cyan] imported"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Importing products", total=None, imported=0)

        with collection.batch.fixed_size(batch_size=1000, concurrent_requests=4) as batch:
            for row in ds:
                if limit > 0 and imported >= limit:
                    break

                if row["price"] is None or row["embeddings"] is None:
                    skipped += 1
                    continue

                props = {
                    "title": row["title"] or "",
                    "description": (row["description"] or "")[:5000],
                    "main_category": row["main_category"] or "",
                    "price": float(row["price"]),
                    "average_rating": float(row["average_rating"] or 0),
                    "rating_number": float(row["rating_number"] or 0),
                    "image": row["image"] or "",
                }
                if row["date_first_available"] is not None:
                    props["date_first_available"] = row["date_first_available"].isoformat() + "Z"

                batch.add_object(properties=props, vector=row["embeddings"])
                imported += 1
                progress.update(task, imported=imported)

    print(f"\nDone: {imported} products imported, {skipped} skipped (no price/embeddings)")

    # Verify
    agg = collection.aggregate.over_all(total_count=True)
    print(f"Collection now has {agg.total_count} objects")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Amazon Products into Weaviate")
    parser.add_argument("--limit", type=int, default=0, help="Max products to import")
    args = parser.parse_args()

    client = weaviate.connect_to_local()
    try:
        create_collection(client)
        import_data(client, args.limit)
    finally:
        client.close()


if __name__ == "__main__":
    main()
