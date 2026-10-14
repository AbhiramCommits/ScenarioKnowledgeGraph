"""SQL tabular baseline for scenario retrieval."""

from pathlib import Path

import pandas as pd
import sqlite3


class SQLBaseline:
    """SQL-based retrieval baseline."""

    def __init__(self, summary_df: pd.DataFrame, db_path: str | Path = ":memory:") -> None:
        """Initialize SQL baseline."""
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))

        summary_df.to_sql("scenarios", self.conn, if_exists="replace", index=False)

        # Create indexes
        self.conn.execute("CREATE INDEX idx_behavior ON scenarios(behavior_label)")
        self.conn.execute("CREATE INDEX idx_min_ttc ON scenarios(min_ttc)")
        self.conn.execute("CREATE INDEX idx_max_speed ON scenarios(max_speed)")
        self.conn.commit()

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search scenarios."""
        # Simple text matching in description
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT scenario_id, behavior_label, min_ttc, max_speed
            FROM scenarios
            WHERE description LIKE ?
            ORDER BY scenario_id
            LIMIT ?
            """,
            (f"%{query}%", k),
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "scenario_id": row[0],
                "behavior_label": row[1],
                "min_ttc": row[2],
                "max_speed": row[3],
                "score": 1.0,
            })

        return results

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
