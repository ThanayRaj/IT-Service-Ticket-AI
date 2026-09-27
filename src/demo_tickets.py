"""Explicitly loaded demonstration ticket text, never labeled predictions."""

import sqlite3


DEMO_TICKET_BODIES = (
    "I cannot connect to the company VPN or open the internal employee portal. "
    "Both worked yesterday, and I need access to submit my timesheet.",
    "I requested a password reset but the email never arrived. I have checked "
    "spam and still cannot sign in to my account.",
    "My card was charged for order 78421, but the order page says it was not placed. "
    "Please check whether the payment went through.",
    "The headphones arrived with a cracked ear cup and will not power on. "
    "I would like help with a replacement or refund.",
    "I would like to return the jacket from my recent order because the size is too small. "
    "Please let me know the return steps.",
    "Could you tell me the current price and available colors for the wireless keyboard?",
    "The customer portal has been unavailable for about twenty minutes. "
    "Our team cannot access order details or submit new requests.",
    "I was charged twice for one purchase. The invoice shows two payments with the same "
    "order reference, and I need help correcting it.",
    "My account is locked after several sign-in attempts, and the unlock link says it expired. "
    "Can you help restore access?",
    "I have a question about the features included with the standard product plan. "
    "Where can I find a comparison with the other options?",
)


class DemoTicketsAlreadyLoaded(RuntimeError):
    """Raised when the demo set is already present and should not be duplicated."""


def load_demo_tickets(application_service) -> dict:
    """Run each sample body once through the normal prediction/intake pipeline.

    Stable source keys make a repeat click idempotent. The lookup also lets a
    later run finish a partially completed batch without duplicating rows.
    """
    database = application_service.database
    created = 0
    skipped = 0
    for index, body in enumerate(DEMO_TICKET_BODIES, start=1):
        source_key = "demo-ticket-{0:02d}".format(index)
        if database.get_ticket_by_source_key(source_key):
            skipped += 1
            continue
        try:
            application_service.submit_ticket(
                body, source="demo", source_key=source_key
            )
            created += 1
        except sqlite3.IntegrityError:
            # A concurrent/repeated request may win the unique-key race after
            # our lookup. Treat that existing ticket as a successful skip.
            if database.get_ticket_by_source_key(source_key):
                skipped += 1
            else:
                raise
    return {"created": created, "skipped": skipped, "total": len(DEMO_TICKET_BODIES)}


def clear_demo_tickets(database) -> int:
    """Remove only records tagged by the explicit demo source."""
    return database.clear_tickets_by_source("demo")
