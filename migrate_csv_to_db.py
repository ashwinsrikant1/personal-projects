"""One-time migration script: load existing CSV data into Lakebase Postgres."""

import csv
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "nutrition_tracker"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=os.getenv("DB_SSLMODE", "require"),
    )


def migrate(csv_path: str = "nutrition_data.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    conn = get_connection()
    cur = conn.cursor()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            cur.execute(
                """INSERT INTO nutrition_entries
                   (profile, date, food_description, calories, protein, carbs, fat, sugar, fiber)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    row["profile"],
                    row["date"],
                    row["food_description"],
                    float(row["calories"]),
                    float(row["protein"]),
                    float(row["carbs"]),
                    float(row["fat"]),
                    float(row["sugar"]),
                    float(row["fiber"]),
                ),
            )
            count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Migrated {count} rows from {csv_path} into nutrition_entries.")


if __name__ == "__main__":
    migrate()
