
""" Online Shopping Delivery & Logistics Analysis"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Looks for the CSV in the same folder as this script (avoids path errors)
DATA_PATH = Path(__file__).resolve().parent / "23_delivery_performance.csv"

# ---------------------------------------------------------------- 1. Load
for enc in ("utf-8", "utf-8-sig", "latin-1"):
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        break
    except UnicodeDecodeError:
        continue
print("=== 1. SHAPE, COLUMNS, DTYPES ===")
print("Shape:", df.shape)
print(df.dtypes, "\n")
print(df.head(), "\n")

# ---------------------------------------------------------------- 2. Missing + duplicates
print("=== 2. MISSING VALUES & DUPLICATES ===")
print(df.isnull().sum(), "\n")
dup_count = df.duplicated(subset="Shipment_ID").sum()
print("Duplicate Shipment_ID rows:", dup_count)
df = df.drop_duplicates(subset="Shipment_ID", keep="first").reset_index(drop=True)
print("Shape after removing duplicates:", df.shape, "\n")

# ---------------------------------------------------------------- 4. Dates (done before step 3, it needs datetimes)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")

# ---------------------------------------------------------------- 3. Handle missing values
print("=== 3. HANDLING MISSING VALUES ===")
# Shipping_Cost -> median per shipping method (robust to outliers, respects method pricing)
df["Shipping_Cost"] = df["Shipping_Cost"].fillna(
    df.groupby("Shipping_Method")["Shipping_Cost"].transform("median")
)
# Delivery_Date -> only for Delivered orders: Order_Date + median duration of that method.
# In Transit / Cancelled orders legitimately have no delivery date, so they stay NaT.
tmp_days = (df["Delivery_Date"] - df["Order_Date"]).dt.days
median_days = tmp_days.groupby(df["Shipping_Method"]).median()
mask = df["Delivery_Date"].isna() & (df["Order_Status"] == "Delivered")
df.loc[mask, "Delivery_Date"] = df.loc[mask, "Order_Date"] + pd.to_timedelta(
    df.loc[mask, "Shipping_Method"].map(median_days), unit="D"
)
print("Delivery_Date imputed for delivered orders:", int(mask.sum()))
print(df.isnull().sum(), "\n")

# ---------------------------------------------------------------- 5. Delivery_Days
df["Delivery_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days
print("=== 5. Delivery_Days created ===")
print(df[["Shipment_ID", "Order_Date", "Delivery_Date", "Delivery_Days"]].head(), "\n")

# ---------------------------------------------------------------- 6. Success %
print("=== 6. DELIVERY SUMMARY ===")
total = len(df)
delivered = int((df["Order_Status"] == "Delivered").sum())
success_pct = delivered / total * 100
print(f"Total shipments: {total}\nDelivered: {delivered}\nSuccess rate: {success_pct:.2f}%\n")

# Analysis dataset: shipments that have a measured delivery duration
dd = df.dropna(subset=["Delivery_Days"]).copy()

# ---------------------------------------------------------------- 7. Avg duration by method
print("=== 7. AVERAGE DELIVERY DAYS BY SHIPPING METHOD ===")
method_avg = dd.groupby("Shipping_Method")["Delivery_Days"].mean().round(2).sort_values()
print(method_avg, "\n")

# ---------------------------------------------------------------- 8. Partner ranking
print("=== 8. DELIVERY PARTNER RANKING ===")
partner_rank = (
    df.groupby("Delivery_Partner")
    .agg(Shipments=("Shipment_ID", "count"))
    .join(dd.groupby("Delivery_Partner")["Delivery_Days"].mean().round(2).rename("Avg_Delivery_Days"))
    .sort_values(["Shipments", "Avg_Delivery_Days"], ascending=[False, True])
)
partner_rank["Rank"] = range(1, len(partner_rank) + 1)
print(partner_rank, "\n")

# ---------------------------------------------------------------- 9. Success by method
print("=== 9. DELIVERY SUCCESS RATE BY SHIPPING METHOD ===")
method_success = (
    df.assign(Is_Delivered=df["Order_Status"].eq("Delivered"))
    .groupby("Shipping_Method")["Is_Delivered"].mean().mul(100).round(2)
    .rename("Success_%").sort_values(ascending=False)
)
print(method_success, "\n")

# ---------------------------------------------------------------- 10. Top 10 longest
print("=== 10. TOP 10 LONGEST DELIVERIES ===")
cols = ["Shipment_ID", "Delivery_Partner", "Shipping_Method", "Order_Date", "Delivery_Date", "Delivery_Days"]
print(dd.nlargest(10, "Delivery_Days")[cols].to_string(index=False), "\n")

# ---------------------------------------------------------------- 11. Dictionary + set
print("=== 11. STATUS COUNTS (dict) & UNIQUE PARTNERS (set) ===")
status_counts = {}
for s in df["Order_Status"]:
    status_counts[s] = status_counts.get(s, 0) + 1
unique_partners = set(df["Delivery_Partner"])
print("Status counts:", status_counts)
print("Unique partners:", unique_partners, "\n")

# ---------------------------------------------------------------- 12. Label function
SLIGHT_LIMIT = 5   # <= 5 days  -> On-Time
HIGH_LIMIT = 8     # 6-8 days   -> Slightly Delayed ; > 8 days -> Highly Delayed


def label_delivery(days):
    """Label a shipment by delivery days."""
    if pd.isna(days):
        return "Not Delivered"
    if days <= SLIGHT_LIMIT:
        return "On-Time"
    if days <= HIGH_LIMIT:
        return "Slightly Delayed"
    return "Highly Delayed"


df["Delay_Label"] = df["Delivery_Days"].apply(label_delivery)
dd["Delay_Label"] = dd["Delivery_Days"].apply(label_delivery)
print("=== 12. DELAY LABELS (On-Time <=5, Slightly 6-8, Highly >8 days) ===")
print(df["Delay_Label"].value_counts(), "\n")

# ---------------------------------------------------------------- 13. NumPy stats
print("=== 13. NUMPY STATISTICS OF DELIVERY_DAYS ===")
arr = dd["Delivery_Days"].to_numpy(dtype=float)
print(f"Mean   : {np.mean(arr):.2f}")
print(f"Median : {np.median(arr):.2f}")
print(f"Std Dev: {np.std(arr):.2f}\n")

# ---------------------------------------------------------------- 14. IQR + monthly
print("=== 14. IQR OUTLIERS & MONTHLY PERFORMANCE ===")
q1, q3 = np.percentile(arr, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1}, Q3={q3}, IQR={iqr}, bounds=({lower}, {upper})")
outliers = dd[(dd["Delivery_Days"] < lower) | (dd["Delivery_Days"] > upper)]
print("Unusually long deliveries:", len(outliers))
print(outliers.sort_values("Delivery_Days", ascending=False)[cols].head(10).to_string(index=False), "\n")

dd["Order_Month"] = dd["Order_Date"].dt.to_period("M").astype(str)
monthly = dd.groupby("Order_Month").agg(
    Shipments=("Shipment_ID", "count"),
    Avg_Delivery_Days=("Delivery_Days", "mean"),
    Pct_Delayed=("Delay_Label", lambda s: (s != "On-Time").mean() * 100),
).round(2)
print(monthly, "\n")

# ---------------------------------------------------------------- 15. Shipment class + recommendations
class Shipment:
    def __init__(self, shipment_id, partner, delivery_days):
        self.shipment_id = shipment_id
        self.partner = partner
        self.delivery_days = delivery_days

    def label(self):
        return label_delivery(self.delivery_days)

    def __repr__(self):
        return f"Shipment({self.shipment_id}, {self.partner}, {self.delivery_days} days, {self.label()})"


print("=== 15. SHIPMENT CLASS DEMO ===")
shipments = [Shipment(r.Shipment_ID, r.Delivery_Partner, r.Delivery_Days)
             for r in dd.nlargest(5, "Delivery_Days").itertuples()]
for s in shipments:
    print(s)

delayed = dd[dd["Delay_Label"] != "On-Time"]
partner_delay = (dd.groupby("Delivery_Partner")["Delay_Label"]
                 .apply(lambda s: (s != "On-Time").mean() * 100).round(1))
worst_partner = partner_delay.idxmax()
best_partner = partner_delay.idxmin()
slowest_method = method_avg.idxmax()
worst_month = monthly["Pct_Delayed"].idxmax()

print("\n=== FIVE RECOMMENDATIONS ===")
recs = [
    f"Shift volume from {worst_partner} ({partner_delay[worst_partner]}% delayed) toward {best_partner} "
    f"({partner_delay[best_partner]}% delayed) and review the SLA with the weakest partner.",
    f"Set realistic delivery promises for '{slowest_method}' shipping (avg {method_avg[slowest_method]} days) "
    "or upgrade its routes.",
    f"Add extra capacity and staffing before peak delay months such as {worst_month}.",
    f"Monitor the {len(outliers)} IQR-outlier shipments weekly, with automatic escalation when a parcel passes {upper:.0f} days.",
    "Track Cancelled/Returned orders and In-Transit parcels with real-time tracking alerts to raise the overall success rate.",
]
for i, r in enumerate(recs, 1):
    print(f"{i}. {r}")

# ---------------------------------------------------------------- Save deliverables
out = Path(__file__).resolve().parent
method_avg.to_csv(out / "out_method_summary.csv")
partner_rank.to_csv(out / "out_partner_ranking.csv")
delayed[cols + ["Delay_Label"]].to_csv(out / "out_delayed_shipments.csv", index=False)
monthly.to_csv(out / "out_monthly_trend.csv")
print("\nSaved: out_method_summary.csv, out_partner_ranking.csv, out_delayed_shipments.csv, out_monthly_trend.csv")