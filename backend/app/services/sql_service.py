import os
import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple

try:
    from backend.app.services.tracing_service import tracing_service
except ImportError:
    try:
        from services.tracing_service import tracing_service
    except ImportError:
        class _DummyTracing:
            def observe(self, *a, **k):
                def d(f):
                    return f
                return d
        tracing_service = _DummyTracing()


FORBIDDEN_SQL_PATTERNS = [
    r"\b(drop|delete|update|insert|alter|create|replace|truncate)\b",
    r"\b(attach|detach|pragma|grant|revoke|vacuum|reindex)\b",
    r";"  # Prevent query stacking
]


class SQLQueryService:
    """
    Structured Text-to-SQL service using local SQLite storage.
    Converts natural-language questions to safe, read-only SQL queries,
    executes them against structured financial/historical data, and synthesizes answers.
    """

    def __init__(self, db_path: Optional[str] = None, model_name: str = "gpt-4o-mini", timeout_seconds: float = 15.0):
        if db_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
            os.makedirs(base_dir, exist_ok=True)
            self.db_path = os.path.join(base_dir, "omnibrain_structured.db")
        else:
            self.db_path = db_path

        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self._init_database()

    def _get_connection(self, read_only: bool = False) -> sqlite3.Connection:
        if read_only and os.path.exists(self.db_path):
            uri_path = f"file:{os.path.abspath(self.db_path)}?mode=ro"
            return sqlite3.connect(uri_path, uri=True)
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Initializes database schema and seeds historical financial metrics table."""
        conn = self._get_connection(read_only=False)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_financials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    quarter TEXT NOT NULL,
                    revenue REAL NOT NULL,
                    net_profit REAL NOT NULL,
                    operating_expenses REAL NOT NULL,
                    gross_margin REAL NOT NULL,
                    headcount INTEGER NOT NULL,
                    notes TEXT
                );
            """)

            cursor.execute("SELECT COUNT(*) FROM company_financials;")
            count = cursor.fetchone()[0]

            if count == 0:
                seed_data = [
                    ("OmniBrain Corp", 2021, "FY", 120.5, 18.2, 85.0, 0.68, 450, "Initial platform scaling"),
                    ("OmniBrain Corp", 2022, "FY", 155.0, 24.5, 102.0, 0.70, 580, "Multi-modal RAG expansion"),
                    ("OmniBrain Corp", 2023, "FY", 198.4, 32.1, 125.5, 0.72, 720, "Enterprise contracts acceleration"),
                    ("OmniBrain Corp", 2024, "FY", 245.8, 41.0, 152.0, 0.73, 890, "LangGraph supervisor architecture integration"),
                    ("OmniBrain Corp", 2025, "FY", 310.2, 55.6, 184.0, 0.75, 1100, "Full autonomous multimodal platform launch"),
                    ("OmniBrain Corp", 2024, "Q1", 56.2, 9.4, 35.0, 0.72, 750, "Q1 FY2024 financial report"),
                    ("OmniBrain Corp", 2024, "Q2", 61.0, 10.2, 38.0, 0.73, 790, "Q2 FY2024 financial report"),
                    ("OmniBrain Corp", 2024, "Q3", 63.8, 10.5, 39.0, 0.73, 830, "Q3 FY2024 financial report"),
                    ("OmniBrain Corp", 2024, "Q4", 64.8, 10.9, 40.0, 0.74, 890, "Q4 FY2024 financial report"),
                    ("OmniBrain Corp", 2025, "Q1", 72.5, 12.8, 43.0, 0.74, 950, "Q1 FY2025 financial report"),
                    ("OmniBrain Corp", 2025, "Q2", 77.0, 13.9, 45.5, 0.75, 1020, "Q2 FY2025 financial report")
                ]
                cursor.executemany("""
                    INSERT INTO company_financials
                    (company, year, quarter, revenue, net_profit, operating_expenses, gross_margin, headcount, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, seed_data)
                conn.commit()
        finally:
            conn.close()

    def get_table_schema(self) -> str:
        """Returns the schema representation for LLM prompt context."""
        return (
            "Table: company_financials\n"
            "Columns:\n"
            "  - id: INTEGER\n"
            "  - company: TEXT (e.g. 'OmniBrain Corp')\n"
            "  - year: INTEGER (2021 to 2025)\n"
            "  - quarter: TEXT ('FY', 'Q1', 'Q2', 'Q3', 'Q4')\n"
            "  - revenue: REAL (in millions USD, e.g. 120.5)\n"
            "  - net_profit: REAL (in millions USD, e.g. 18.2)\n"
            "  - operating_expenses: REAL (in millions USD)\n"
            "  - gross_margin: REAL (fraction, e.g. 0.75 for 75%)\n"
            "  - headcount: INTEGER\n"
            "  - notes: TEXT"
        )

    def is_query_safe(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates that SQL statement is strictly read-only and free of destructive clauses.
        """
        clean_sql = sql.strip().rstrip(";").strip()
        if not (clean_sql.lower().startswith("select") or clean_sql.lower().startswith("with")):
            return False, "Query blocked: Only SELECT statements are permitted."

        for pattern in FORBIDDEN_SQL_PATTERNS:
            if re.search(pattern, clean_sql, re.IGNORECASE):
                return False, f"Query blocked: Destructive or unauthorized SQL pattern detected ('{pattern}')."

        return True, None

    @tracing_service.observe(name="sql_generate_query", as_type="generation")
    def generate_sql(self, question: str) -> str:
        """
        Converts natural-language question to an executable SQLite query.
        Uses OpenAI when configured, else falls back to domain heuristic templates.
        """
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                prompt = (
                    "You are OmniBrain SQL Agent. Convert the user question into a single valid SQLite SELECT query.\n"
                    f"{self.get_table_schema()}\n\n"
                    "Rules:\n"
                    "1. Return ONLY the raw SQL query. Do not wrap in markdown or backticks.\n"
                    "2. Always query from 'company_financials'.\n"
                    "3. When filtering by annual data, filter by quarter = 'FY' unless a specific quarter is requested.\n"
                    f"Question: {question}"
                )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a specialized Text-to-SQL converter for SQLite."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=150
                )
                raw_sql = response.choices[0].message.content.strip()
                # Strip potential markdown fences
                raw_sql = re.sub(r"^```(sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
                raw_sql = re.sub(r"```$", "", raw_sql).strip()
                return raw_sql.rstrip(";")
            except Exception as e:
                print(f"[SQLQueryService] LLM SQL generation error: {e}, falling back to template")

        # Offline / Heuristic Fallback
        q = question.lower()
        if "profit" in q:
            match_year = re.search(r"\b(202[0-9])\b", q)
            if match_year:
                year = int(match_year.group(1))
                return f"SELECT year, quarter, net_profit, revenue FROM company_financials WHERE year = {year} AND quarter = 'FY'"
            return "SELECT year, quarter, net_profit FROM company_financials WHERE quarter = 'FY' ORDER BY year ASC"

        if "headcount" in q:
            return "SELECT year, quarter, headcount FROM company_financials WHERE quarter = 'FY' ORDER BY year ASC"

        if "expense" in q or "operating" in q:
            return "SELECT year, quarter, operating_expenses FROM company_financials WHERE quarter = 'FY' ORDER BY year ASC"

        if "margin" in q:
            return "SELECT year, quarter, gross_margin FROM company_financials WHERE quarter = 'FY' ORDER BY year ASC"

        # General revenue / trend / historical comparison
        return "SELECT year, quarter, revenue, net_profit FROM company_financials WHERE quarter = 'FY' ORDER BY year ASC"

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Safely executes read-only SQL query against local SQLite database.
        """
        is_safe, error_msg = self.is_query_safe(sql)
        if not is_safe:
            return {
                "success": False,
                "error": error_msg,
                "columns": [],
                "rows": [],
                "row_count": 0
            }

        try:
            conn = self._get_connection(read_only=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            formatted_rows = [dict(row) for row in rows]
            conn.close()

            return {
                "success": True,
                "columns": columns,
                "rows": formatted_rows,
                "row_count": len(formatted_rows),
                "error": None
            }
        except sqlite3.Error as e:
            print(f"[SQLQueryService] SQLite execution error: {e}")
            return {
                "success": False,
                "error": f"SQL syntax or execution error: {str(e)}",
                "columns": [],
                "rows": [],
                "row_count": 0
            }

    @tracing_service.observe(name="sql_synthesize_answer", as_type="generation")
    def synthesize_answer(self, question: str, sql: str, query_result: Dict[str, Any]) -> str:
        """
        Synthesizes natural-language answer from executed SQL and resulting rows.
        """
        if not query_result.get("success"):
            return f"Unable to execute SQL query: {query_result.get('error')}"

        rows = query_result.get("rows", [])
        if not rows:
            return f"SQL query executed successfully (`{sql}`), but returned 0 records."

        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                prompt = (
                    "You are OmniBrain SQL Agent. Formulate a concise, factual answer to the user's question "
                    "using the structured query results below. Include exact figures, units (millions USD), and years.\n\n"
                    f"User Question: {question}\n"
                    f"Executed SQL: {sql}\n"
                    f"Results: {rows}\n\n"
                    "Answer:"
                )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a quantitative financial analyst answering from structured SQLite data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=250
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[SQLQueryService] LLM synthesis error: {e}")

        # Structured Offline Answer Formulation
        lines = [f"**Structured SQL Query Results** (`{sql}`):"]
        for row in rows:
            row_parts = []
            for k, v in row.items():
                if isinstance(v, float) and "margin" in k:
                    row_parts.append(f"{k}: {v*100:.1f}%")
                elif isinstance(v, float):
                    row_parts.append(f"{k}: ${v:.1f}M")
                else:
                    row_parts.append(f"{k}: {v}")
            lines.append(f"- {', '.join(row_parts)}")

        return "\n".join(lines)

    @tracing_service.observe(name="sql_agent_query")
    def query(self, question: str) -> Dict[str, Any]:
        """
        End-to-end Text-to-SQL entry point: generates SQL, executes safely, and synthesizes answer.
        """
        sql = self.generate_sql(question)
        result = self.execute_sql(sql)
        answer = self.synthesize_answer(question, sql, result)

        return {
            "question": question,
            "executed_sql": sql,
            "success": result["success"],
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": result["row_count"],
            "error": result["error"],
            "answer": answer,
            "status": "COMPLETED" if result["success"] else "SQL_ERROR"
        }
