"""
tools/tools.py — All tools available to the agent.

HUMAN ANALOGY:
Tools are like the apps on your phone. Claude is the smart person holding
the phone. Claude decides WHICH app to open based on what you need.
The calculator app does maths. The maps app does directions. Claude doesn't
do those things itself — it calls the app and reads the result.

TOOL DESIGN PRINCIPLES:
1. Each tool does ONE thing (Single Responsibility)
2. Tools return strings (Claude reads text, not objects)
3. Tools NEVER raise exceptions to Claude — return error strings instead
4. Tool descriptions must be specific enough for Claude to know WHEN to use them
5. Tool input schemas must be strict — this prevents Claude from calling with wrong params

FAILURE SCENARIO — TOOL AMBIGUITY:
If two tools have similar descriptions, Claude picks randomly.
Example: "get_price" and "fetch_stock_price" — too similar, Claude gets confused.
Fix: make names and descriptions clearly distinct.
"""

import math
import json
import random
from datetime import datetime, timedelta
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (the schema Claude sees)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "calculate",
        "description": (
            "Perform mathematical calculations: arithmetic, percentages, compound interest, "
            "EMI calculations, tax calculations, investment returns. "
            "Use for ANY numerical computation. Do NOT calculate in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression. Examples: '100 * 1.18', '50000 * (1.12 ** 5)', '(10000 * 0.08 * 3)'"
                },
                "description": {
                    "type": "string",
                    "description": "What this calculation represents. Example: 'GST on Rs 100'"
                }
            },
            "required": ["expression", "description"]
        }
    },
    {
        "name": "get_stock_price",
        "description": (
            "Get current stock price, 52-week high/low, P/E ratio, and market cap for "
            "Indian stocks (NSE/BSE). Use ticker symbols like RELIANCE, TCS, INFY, HDFC."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "NSE ticker symbol. Examples: RELIANCE, TCS, INFY, HDFCBANK, WIPRO"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "track_order",
        "description": (
            "Track the current status and location of a customer order. "
            "Returns status, estimated delivery date, and tracking history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID. Format: ORD-XXXXXX (e.g. ORD-123456)"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "calculate_emi",
        "description": (
            "Calculate monthly EMI for a loan given principal amount, annual interest rate, "
            "and tenure in months. Returns EMI amount, total payment, and total interest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "Loan amount in rupees"
                },
                "annual_rate": {
                    "type": "number",
                    "description": "Annual interest rate as percentage. Example: 8.5 for 8.5%"
                },
                "tenure_months": {
                    "type": "integer",
                    "description": "Loan tenure in months. Example: 60 for 5 years"
                }
            },
            "required": ["principal", "annual_rate", "tenure_months"]
        }
    },
    {
        "name": "get_entertainment_info",
        "description": (
            "Get current ratings, available platforms, cast, and genre information "
            "for movies, TV shows, or OTT series in India."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Movie or show title"
                },
                "type": {
                    "type": "string",
                    "enum": ["movie", "show", "both"],
                    "description": "Content type to search"
                }
            },
            "required": ["title", "type"]
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate(expression: str, description: str) -> str:
    """
    Safe mathematical calculator.

    SECURITY NOTE:
    We use eval() here for demo purposes only. In production, use a proper
    math parser library like 'numexpr' or 'simpleeval' to prevent code injection.
    Never eval() user input directly.

    FAILURE SCENARIO:
    User asks "what is 100/0?" — ZeroDivisionError.
    We catch ALL exceptions and return error strings so Claude can
    tell the user gracefully instead of the agent crashing.
    """
    # Safe math environment — only allow math operations, no builtins
    safe_env = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
        "round": round,
        "log": math.log,
    }
    try:
        result = eval(expression, safe_env)
        return f"{description}: {result:,.2f}"
    except ZeroDivisionError:
        return f"Error: Cannot divide by zero in '{expression}'"
    except Exception as e:
        return f"Calculation error for '{expression}': {str(e)}"


def get_stock_price(ticker: str) -> str:
    """
    Simulated stock price lookup.

    In production: call NSE API, Yahoo Finance API, or Alpha Vantage.
    We simulate realistic data here for demo purposes.

    FAILURE SCENARIO — STALE DATA:
    Stock APIs sometimes return cached/stale data. Always include the
    timestamp in the response so Claude can warn users about data freshness.
    """
    # Simulated stock data (realistic Indian stock prices)
    stocks = {
        "RELIANCE":  {"price": 2847.50, "change": +1.2,  "pe": 28.4, "high52": 3024.90, "low52": 2220.30},
        "TCS":       {"price": 3912.00, "change": -0.4,  "pe": 31.2, "high52": 4255.00, "low52": 3311.00},
        "INFY":      {"price": 1567.80, "change": +0.8,  "pe": 24.6, "high52": 1903.00, "low52": 1351.00},
        "HDFCBANK":  {"price": 1642.30, "change": -1.1,  "pe": 19.8, "high52": 1880.00, "low52": 1363.45},
        "WIPRO":     {"price": 298.45,  "change": +2.1,  "pe": 21.3, "high52": 320.00,  "low52": 205.00},
        "ADANIPORTS":{"price": 1156.00, "change": +0.3,  "pe": 22.1, "high52": 1608.00, "low52": 915.00},
    }

    ticker_upper = ticker.upper().strip()
    if ticker_upper not in stocks:
        return f"Stock '{ticker}' not found. Available: {', '.join(stocks.keys())}"

    s = stocks[ticker_upper]
    direction = "▲" if s["change"] > 0 else "▼"
    return (
        f"{ticker_upper} | Price: ₹{s['price']:,.2f} {direction}{abs(s['change'])}% today | "
        f"P/E: {s['pe']} | 52W High: ₹{s['high52']:,} | 52W Low: ₹{s['low52']:,} | "
        f"Data as of {datetime.now().strftime('%d %b %Y %H:%M')} IST"
    )


def track_order(order_id: str) -> str:
    """
    Simulated order tracking system.

    FAILURE SCENARIO — ORDER NOT FOUND:
    Instead of returning None or raising an exception (which would crash the agent),
    we return a helpful error string. Claude then tells the user politely.

    FAILURE SCENARIO — PARTIAL ORDER ID:
    User types "123456" instead of "ORD-123456". We handle both formats.
    """
    # Normalize order ID format
    normalized = order_id.upper().strip()
    if not normalized.startswith("ORD-"):
        normalized = f"ORD-{normalized}"

    # Simulated order database
    orders = {
        "ORD-123456": {
            "status": "OUT_FOR_DELIVERY",
            "item": "Dell XPS 15 Laptop",
            "ordered": "2026-02-22",
            "eta": "Today by 8:00 PM",
            "location": "Hyderabad Hub → Local Delivery Partner",
            "history": [
                ("2026-02-22 10:30", "Order placed"),
                ("2026-02-22 14:00", "Payment confirmed"),
                ("2026-02-23 09:00", "Picked from warehouse"),
                ("2026-02-24 16:00", "Reached Hyderabad hub"),
                ("2026-02-25 08:30", "Out for delivery"),
            ]
        },
        "ORD-789012": {
            "status": "PROCESSING",
            "item": "Samsung S24 Phone",
            "ordered": "2026-02-26",
            "eta": "3-5 business days",
            "location": "Warehouse — Bengaluru",
            "history": [
                ("2026-02-26 11:00", "Order placed"),
                ("2026-02-26 11:05", "Payment confirmed"),
            ]
        },
    }

    if normalized not in orders:
        return (
            f"Order '{normalized}' not found. "
            "Please check your order ID (format: ORD-XXXXXX). "
            "You can find it in your confirmation email."
        )

    o = orders[normalized]
    history_str = "\n".join(f"  {t}: {e}" for t, e in o["history"])
    return (
        f"Order {normalized} | Status: {o['status']}\n"
        f"Item: {o['item']}\n"
        f"Estimated Delivery: {o['eta']}\n"
        f"Current Location: {o['location']}\n"
        f"History:\n{history_str}"
    )


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """
    Calculate loan EMI using the standard formula.
    EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    where r = monthly rate, n = tenure in months

    HUMAN ANALOGY:
    Like a bank's loan calculator. You give it the loan amount, interest rate,
    and how many months you want to pay. It tells you your monthly payment.
    """
    if principal <= 0:
        return "Error: Principal must be positive"
    if annual_rate <= 0 or annual_rate > 50:
        return "Error: Interest rate must be between 0 and 50%"
    if tenure_months <= 0:
        return "Error: Tenure must be positive"

    r = annual_rate / (12 * 100)  # Monthly interest rate
    n = tenure_months

    if r == 0:  # Zero interest
        emi = principal / n
    else:
        emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)

    total_payment = emi * n
    total_interest = total_payment - principal

    return (
        f"Loan: ₹{principal:,.0f} | Rate: {annual_rate}% | Tenure: {tenure_months} months\n"
        f"Monthly EMI: ₹{emi:,.2f}\n"
        f"Total Payment: ₹{total_payment:,.2f}\n"
        f"Total Interest: ₹{total_interest:,.2f} ({total_interest/principal*100:.1f}% of principal)"
    )


def get_entertainment_info(title: str, type: str) -> str:
    """
    Simulated entertainment database lookup.

    In production: call TMDB API, IMDB API, or JustWatch API.
    """
    catalog = {
        "scam 1992":     {"rating": 9.5, "platform": "SonyLIV", "genre": "Crime Drama", "episodes": 10, "year": 2020},
        "panchayat":     {"rating": 9.0, "platform": "Prime Video", "genre": "Comedy Drama", "episodes": 24, "year": 2020},
        "mirzapur":      {"rating": 8.5, "platform": "Prime Video", "genre": "Crime Thriller", "episodes": 28, "year": 2018},
        "the family man": {"rating": 9.0, "platform": "Prime Video", "genre": "Action Thriller", "episodes": 18, "year": 2019},
        "sacred games":  {"rating": 8.7, "platform": "Netflix", "genre": "Crime Thriller", "episodes": 16, "year": 2018},
        "rrr":           {"rating": 8.0, "platform": "Netflix/ZEE5", "genre": "Action Drama", "year": 2022},
        "kgf":           {"rating": 8.4, "platform": "Prime Video", "genre": "Action", "year": 2018},
        "delhi crime":   {"rating": 8.5, "platform": "Netflix", "genre": "Crime Drama", "episodes": 14, "year": 2019},
    }

    key = title.lower().strip()
    if key not in catalog:
        return (
            f"'{title}' not found in catalog. "
            f"Popular titles available: {', '.join(list(catalog.keys())[:5])}"
        )

    c = catalog[key]
    episodes_str = f" | {c['episodes']} episodes" if "episodes" in c else " | Movie"
    return (
        f"{title.title()} | Rating: {c['rating']}/10 | {c['genre']} | "
        f"Platform: {c['platform']}{episodes_str} | Year: {c['year']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "calculate": calculate,
    "get_stock_price": get_stock_price,
    "track_order": track_order,
    "calculate_emi": calculate_emi,
    "get_entertainment_info": get_entertainment_info,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Central tool dispatcher with error handling and retry logic.

    FAILURE SCENARIO — WRONG TOOL INPUT:
    Claude sometimes passes wrong parameter types (e.g. "8.5%" instead of 8.5).
    We catch TypeError and return a clear error so Claude can retry with fixed params.

    FAILURE SCENARIO — TOOL NOT FOUND:
    Claude hallucinates a tool name that doesn't exist.
    We return a clear error listing available tools.
    """
    if tool_name not in TOOL_MAP:
        available = ", ".join(TOOL_MAP.keys())
        return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

    tool_fn = TOOL_MAP[tool_name]
    try:
        result = tool_fn(**tool_input)
        return str(result)
    except TypeError as e:
        return f"Tool '{tool_name}' called with wrong parameters: {e}. Input was: {tool_input}"
    except Exception as e:
        return f"Tool '{tool_name}' failed unexpectedly: {e}"
