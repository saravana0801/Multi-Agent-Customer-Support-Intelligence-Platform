"""
Generate synthetic PostgreSQL OMS (Order Management System) data.
Uses Pydantic models to validate records and outputs:
1. data/mock_oms_orders.json
2. data/init_oms_postgres.sql (PostgreSQL DDL + INSERT statements)
"""

import csv
import json
import random
import re
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    product_name: str
    product_segment: str
    unit_price: float
    quantity: int = 1


class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    customer_name: str
    customer_email: str
    order_date: str
    delivery_date: Optional[str] = None
    status: str  # DELIVERED, IN_TRANSIT, SHIPPED, RETURN_REQUESTED, REFUNDED
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    items: List[OrderItem]
    total_amount: float
    shipping_address: str


def generate_data():
    print("Reading support_tickets_10k.csv for product and order references...")
    products = []
    order_ids = []

    with open("data/support_tickets_10k.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_name = row.get("product_name")
            p_seg = row.get("product_segment")
            if p_name and p_seg:
                products.append((p_name, p_seg))

            text = row.get("ticket_text", "")
            match = re.search(r"#?ORD(\d{5,8})", text, re.IGNORECASE)
            if match:
                order_ids.append(f"ORD{match.group(1)}")

    # Deduplicate products
    unique_products = list(set(products))
    unique_order_ids = list(set(order_ids))[:300]
    print(f"Found {len(unique_products)} unique products and {len(unique_order_ids)} referenced Order IDs.")

    # Complement order IDs up to 500
    while len(unique_order_ids) < 500:
        unique_order_ids.append(f"ORD{random.randint(1000000, 9999999)}")

    carriers = ["FedEx", "UPS", "DHL", "BlueDart", "USPS"]
    statuses = ["DELIVERED", "IN_TRANSIT", "SHIPPED", "RETURN_REQUESTED", "REFUNDED", "DELIVERED", "DELIVERED"]
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Avery"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez"]
    streets = ["Oak St", "Pine Ave", "Maple Dr", "Cedar Blvd", "Main St", "Elm St", "Washington Ave"]
    cities = ["New York, NY", "Austin, TX", "Seattle, WA", "Chicago, IL", "San Francisco, CA", "Miami, FL"]

    orders: List[OrderRecord] = []

    base_date = datetime(2023, 8, 1)

    for i, oid in enumerate(unique_order_ids):
        c_first = random.choice(first_names)
        c_last = random.choice(last_names)
        c_name = f"{c_first} {c_last}"
        c_id = f"CUST-{1000 + i}"
        c_email = f"{c_first.lower()}.{c_last.lower()}{random.randint(10, 99)}@example.com"

        days_ago = random.randint(1, 30)
        order_dt = base_date - timedelta(days=days_ago)
        order_date_str = order_dt.strftime("%Y-%m-%d")

        status = random.choice(statuses)
        carrier = random.choice(carriers)
        trk = f"{carrier[:3].upper()}-{random.randint(10000000, 99999999)}"

        if status in ["DELIVERED", "RETURN_REQUESTED", "REFUNDED"]:
            delivery_dt = order_dt + timedelta(days=random.randint(2, 5))
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d")
        else:
            delivery_date_str = None

        # 1-3 items
        num_items = random.randint(1, 3)
        order_items = []
        total = 0.0
        for _ in range(num_items):
            p = random.choice(unique_products)
            price = round(random.uniform(15.0, 350.0), 2)
            qty = random.randint(1, 2)
            order_items.append(OrderItem(
                product_name=p[0],
                product_segment=p[1],
                unit_price=price,
                quantity=qty
            ))
            total += price * qty

        addr = f"{random.randint(100, 999)} {random.choice(streets)}, {random.choice(cities)} {random.randint(10000, 99999)}"

        record = OrderRecord(
            order_id=oid,
            customer_id=c_id,
            customer_name=c_name,
            customer_email=c_email,
            order_date=order_date_str,
            delivery_date=delivery_date_str,
            status=status,
            carrier=carrier,
            tracking_number=trk,
            items=order_items,
            total_amount=round(total, 2),
            shipping_address=addr
        )
        orders.append(record)

    # 1. Save JSON
    json_path = "data/mock_oms_orders.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([o.model_dump() for o in orders], f, indent=2)
    print(f"Generated {len(orders)} synthetic OMS orders in {json_path}")

    # 2. Save PostgreSQL SQL Script
    sql_path = "data/init_oms_postgres.sql"
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("-- PostgreSQL Schema & Seed Data for E-Commerce OMS Database\n\n")
        f.write("CREATE TABLE IF NOT EXISTS orders (\n")
        f.write("    order_id VARCHAR(50) PRIMARY KEY,\n")
        f.write("    customer_id VARCHAR(50) NOT NULL,\n")
        f.write("    customer_name VARCHAR(100) NOT NULL,\n")
        f.write("    customer_email VARCHAR(100) NOT NULL,\n")
        f.write("    order_date DATE NOT NULL,\n")
        f.write("    delivery_date DATE,\n")
        f.write("    status VARCHAR(50) NOT NULL,\n")
        f.write("    carrier VARCHAR(50),\n")
        f.write("    tracking_number VARCHAR(100),\n")
        f.write("    items JSONB NOT NULL,\n")
        f.write("    total_amount NUMERIC(10, 2) NOT NULL,\n")
        f.write("    shipping_address TEXT NOT NULL,\n")
        f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n")
        f.write(");\n\n")
        f.write("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);\n")
        f.write("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);\n\n")

        for o in orders:
            items_json = json.dumps([item.model_dump() for item in o.items]).replace("'", "''")
            del_val = f"'{o.delivery_date}'" if o.delivery_date else "NULL"
            f.write(
                f"INSERT INTO orders (order_id, customer_id, customer_name, customer_email, order_date, delivery_date, status, carrier, tracking_number, items, total_amount, shipping_address) "
                f"VALUES ('{o.order_id}', '{o.customer_id}', '{o.customer_name}', '{o.customer_email}', '{o.order_date}', {del_val}, '{o.status}', '{o.carrier}', '{o.tracking_number}', '{items_json}'::jsonb, {o.total_amount}, '{o.shipping_address}') "
                f"ON CONFLICT (order_id) DO NOTHING;\n"
            )
    print(f"Generated PostgreSQL DDL & Seed statements in {sql_path}")


if __name__ == "__main__":
    generate_data()
