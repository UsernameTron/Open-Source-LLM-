"""Statistical reference database for training."""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import aiosqlite

logger = logging.getLogger(__name__)

@dataclass
class StatisticEntry:
    """Represents a statistical data point."""
    value: float
    source: str
    category: str
    timestamp: datetime
    confidence: float
    context: str
    metadata: Dict[str, Any]

class StatisticalDatabase:
    """Manages curated statistics for model training."""
    
    def __init__(self, db_path: str = "stats.db"):
        """Initialize database."""
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY,
                    value REAL NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    confidence REAL NOT NULL,
                    context TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category
                ON statistics(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON statistics(source)
            """)
            
    async def add_statistic(self, entry: StatisticEntry):
        """Add a new statistical entry."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO statistics
                    (value, source, category, timestamp,
                     confidence, context, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.value,
                        entry.source,
                        entry.category,
                        entry.timestamp.isoformat(),
                        entry.confidence,
                        entry.context,
                        json.dumps(entry.metadata)
                    )
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error adding statistic: {str(e)}")
            raise
            
    async def get_statistics(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[StatisticEntry]:
        """Retrieve statistics with optional filtering."""
        try:
            query = """
                SELECT value, source, category, timestamp,
                       confidence, context, metadata
                FROM statistics
                WHERE confidence >= ?
            """
            params = [min_confidence]
            
            if category:
                query += " AND category = ?"
                params.append(category)
                
            if source:
                query += " AND source = ?"
                params.append(source)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
            return [
                StatisticEntry(
                    value=row[0],
                    source=row[1],
                    category=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    confidence=row[4],
                    context=row[5],
                    metadata=json.loads(row[6])
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {str(e)}")
            raise
            
    async def update_confidence(
        self,
        statistic_id: int,
        new_confidence: float
    ):
        """Update confidence score for a statistic."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE statistics
                    SET confidence = ?
                    WHERE id = ?
                    """,
                    (new_confidence, statistic_id)
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating confidence: {str(e)}")
            raise
            
    async def remove_outdated(
        self,
        max_age_days: int = 365,
        min_confidence: float = 0.7
    ):
        """Remove outdated statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    DELETE FROM statistics
                    WHERE julianday('now') - julianday(timestamp) > ?
                    AND confidence < ?
                    """,
                    (max_age_days, min_confidence)
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error removing outdated stats: {str(e)}")
            raise
            
    async def get_categories(self) -> List[str]:
        """Get list of all categories."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT DISTINCT category FROM statistics"
                ) as cursor:
                    rows = await cursor.fetchall()
                    
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            raise
            
    async def get_sources(self) -> List[str]:
        """Get list of all sources."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT DISTINCT source FROM statistics"
                ) as cursor:
                    rows = await cursor.fetchall()
                    
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            raise
            
    async def export_json(self, filepath: str):
        """Export database to JSON file."""
        try:
            stats = await self.get_statistics(limit=1000000)
            data = [
                {
                    "value": stat.value,
                    "source": stat.source,
                    "category": stat.category,
                    "timestamp": stat.timestamp.isoformat(),
                    "confidence": stat.confidence,
                    "context": stat.context,
                    "metadata": stat.metadata
                }
                for stat in stats
            ]
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error exporting database: {str(e)}")
            raise
            
    async def import_json(self, filepath: str):
        """Import database from JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            for item in data:
                entry = StatisticEntry(
                    value=item["value"],
                    source=item["source"],
                    category=item["category"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    confidence=item["confidence"],
                    context=item["context"],
                    metadata=item["metadata"]
                )
                await self.add_statistic(entry)
                
        except Exception as e:
            logger.error(f"Error importing database: {str(e)}")
            raise
