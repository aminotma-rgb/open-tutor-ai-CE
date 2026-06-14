"""Tool — evaluate SELECT SQL on an in-memory SQLite DB with W3Schools sample tables."""

from __future__ import annotations

import sqlite3

from langchain_core.tools import tool

_SCHEMA_AND_DATA = """
CREATE TABLE Customers (CustomerID INT, CustomerName TEXT, Country TEXT, City TEXT);
CREATE TABLE Products  (ProductID INT, ProductName TEXT, Price REAL, Category TEXT);
CREATE TABLE Orders    (OrderID INT, CustomerID INT, ProductID INT, Quantity INT, OrderDate TEXT);
CREATE TABLE Employees (EmployeeID INT, LastName TEXT, FirstName TEXT, Title TEXT);

INSERT INTO Customers VALUES (1,'Alice','France','Paris'),(2,'Bob','Germany','Berlin'),(3,'Charlie','UK','London');
INSERT INTO Products  VALUES (1,'Laptop',999.99,'Electronics'),(2,'Phone',599.99,'Electronics'),(3,'Desk',299.99,'Furniture');
INSERT INTO Orders    VALUES (1,1,1,2,'2024-01-10'),(2,2,2,1,'2024-01-11'),(3,1,3,1,'2024-01-12');
INSERT INTO Employees VALUES (1,'Smith','John','Manager'),(2,'Doe','Jane','Developer'),(3,'Brown','Bob','Analyst');
"""

_BLOCKED_PREFIXES = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "ATTACH",
    "PRAGMA",
)


@tool
def sql_evaluator(query: str) -> str:
    """Execute a SQL SELECT query on sample W3Schools tables (Customers, Products, Orders, Employees).

    Use this to validate a SQL exercise or to show the result of a learner's query.
    Only SELECT, WITH and EXPLAIN statements are allowed — no INSERT/UPDATE/DROP.
    Pass only the raw SQL string, no markdown fences.

    Available tables: Customers(CustomerID, CustomerName, Country, City),
                      Products(ProductID, ProductName, Price, Category),
                      Orders(OrderID, CustomerID, ProductID, Quantity, OrderDate),
                      Employees(EmployeeID, LastName, FirstName, Title)

    Example:
      sql_evaluator("SELECT CustomerName, Country FROM Customers WHERE Country='France'")
    """
    clean = query.strip()
    upper = clean.upper()
    if any(upper.startswith(p) for p in _BLOCKED_PREFIXES):
        return "Only SELECT, WITH and EXPLAIN statements are allowed."
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(_SCHEMA_AND_DATA)
        cur = conn.execute(clean)
        rows = cur.fetchmany(20)
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        if not rows:
            return "Query returned no results."
        header = " | ".join(cols)
        sep = "-" * len(header)
        lines = [header, sep] + [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    except Exception as exc:
        return f"SQL error: {exc}"
