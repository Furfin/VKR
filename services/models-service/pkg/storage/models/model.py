import psycopg2
import json
from psycopg2 import sql
from psycopg2.extras import DictCursor
from psycopg2.extras import Json
from typing import Dict, Any, Optional, List
import contextlib

class ModelsStorageClient:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.table_name = "models"
        self._initialize_table()

    @contextlib.contextmanager
    def _get_cursor(self):
        """Context manager for database connection handling."""
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                yield cursor
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_table(self):
        create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) UNIQUE NOT NULL,
            data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_name ON {table} (model_name);
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(create_table_query)

    def create_object(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        query = sql.SQL("""
        INSERT INTO {table} (model_name, data)
        VALUES (%s, %s)
        RETURNING id, model_name, data, created_at, updated_at;
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(query, (model_name, json.dumps(data)))
            result = cursor.fetchone()
            return dict(result)

    def get_object_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        query = sql.SQL("""
        SELECT data
        FROM {table}
        WHERE model_name = %s;
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def update_object(self, name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = sql.SQL("""
        UPDATE {table}
        SET data = %s, updated_at = CURRENT_TIMESTAMP
        WHERE model_name = %s
        RETURNING id, model_name, data, created_at, updated_at;
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(query, (Json(data), name))
            result = cursor.fetchone()
            return dict(result) if result else None

    def delete_object(self, name: str) -> bool:
        query = sql.SQL("""
        DELETE FROM {table}
        WHERE model_name = %s
        RETURNING id;
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(query, (name,))
            return cursor.fetchone() is not None

    def list_objects(self, limit: int = 100) -> List[Dict[str, Any]]:
        query = sql.SQL("""
        SELECT id, model_name, data, created_at, updated_at
        FROM {table}
        ORDER BY created_at DESC
        LIMIT %s;
        """).format(table=sql.Identifier(self.table_name))
        
        with self._get_cursor() as cursor:
            cursor.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]