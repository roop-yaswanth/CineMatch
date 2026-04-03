#!/usr/bin/env python3
"""
admin_cleanup.py — Delete a CineMatch user by email.

Usage:
    python admin_cleanup.py yash232n@gmail.com
    python admin_cleanup.py --list                  # List all users
    python admin_cleanup.py --purge-anon            # Delete all anonymous users

Requires CINEMATCH_MONGO_URI environment variable.
"""
import os
import sys
from datetime import datetime, timezone

try:
    from pymongo import MongoClient
except ImportError:
    print("Install pymongo: pip install pymongo")
    sys.exit(1)

MONGO_URI = os.getenv("CINEMATCH_MONGO_URI")
MONGO_DB = os.getenv("CINEMATCH_MONGO_DB", "Cinimatch")

if not MONGO_URI:
    print("Set CINEMATCH_MONGO_URI environment variable.")
    sys.exit(1)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]


def list_users():
    users = list(db.users.find({}, {"_id": 0, "user_id": 1, "identifier": 1, "interaction_count": 1, "updated_at": 1}).sort("updated_at", -1))
    print(f"\n{'Email/Identifier':<35} {'User ID':<15} {'Interactions':<15} {'Last Active'}")
    print("─" * 90)
    for u in users:
        ident = u.get("identifier", "—")
        uid = u.get("user_id", "?")
        count = u.get("interaction_count", 0)
        updated = u.get("updated_at", "")
        print(f"{ident:<35} {uid:<15} {count:<15} {updated}")
    print(f"\nTotal: {len(users)} users")


def delete_user(email: str):
    email = email.strip().lower()
    users = list(db.users.find({"identifier": email}))
    if not users:
        print(f"No user found with identifier '{email}'.")
        return

    user_ids = [u["user_id"] for u in users]
    print(f"\nFound {len(users)} user doc(s) for '{email}':")
    for u in users:
        print(f"  user_id={u['user_id']}, interactions={u.get('interaction_count', 0)}")

    confirm = input("\nDelete ALL data for this user? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    for uid in user_ids:
        r1 = db.users.delete_many({"user_id": uid})
        r2 = db.sessions.delete_many({"user_id": uid})
        r3 = db.interactions.delete_many({"user_id": uid})
        print(f"  Deleted user_id={uid}: {r1.deleted_count} users, {r2.deleted_count} sessions, {r3.deleted_count} interactions")

    print("Done.")


def purge_anonymous():
    """Delete all users without an identifier (anonymous/test users)."""
    anon_users = list(db.users.find({"$or": [{"identifier": {"$exists": False}}, {"identifier": ""}]}))
    if not anon_users:
        print("No anonymous users found.")
        return

    print(f"\nFound {len(anon_users)} anonymous user(s).")
    confirm = input("Delete ALL anonymous users? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    for u in anon_users:
        uid = u["user_id"]
        db.users.delete_many({"user_id": uid})
        db.sessions.delete_many({"user_id": uid})
        db.interactions.delete_many({"user_id": uid})

    print(f"Purged {len(anon_users)} anonymous users and their data.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    arg = sys.argv[1]
    if arg == "--list":
        list_users()
    elif arg == "--purge-anon":
        purge_anonymous()
    else:
        delete_user(arg)
