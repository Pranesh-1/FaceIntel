"""
Visitor Counter Service
Derives and maintains the count of unique visitors from registered faces.
"""

import logging

logger = logging.getLogger(__name__)


class VisitorCounter:
    def __init__(self, db):
        """
        Args:
            db: Database instance – used to retrieve the persisted count.
        """
        self.db = db
        # In-memory counter kept in sync with the DB for O(1) reads
        self._count: int = self.db.count_unique_visitors()
        logger.info(f"Visitor counter initialised at {self._count} unique visitor(s).")

    def increment(self):
        """Called each time a brand-new face is registered."""
        self._count += 1
        logger.info(f"Unique visitor count → {self._count}")

    @property
    def count(self) -> int:
        return self._count

    def sync_from_db(self):
        """Re-sync the in-memory counter from the database (useful after restarts)."""
        self._count = self.db.count_unique_visitors()
        logger.info(f"Synced visitor count from DB: {self._count}")
