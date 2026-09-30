import os
from functools import wraps

from dotenv import load_dotenv
from flask import redirect, session, url_for
from werkzeug.security import check_password_hash

load_dotenv()


def is_admin_authenticated() -> bool:
    return bool(session.get("is_admin"))


def verify_admin_password(password: str) -> bool:
    password_hash = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
    if not password_hash or not password:
        return False
    return check_password_hash(password_hash, password)


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_admin_authenticated():
            return redirect(url_for("login", next=url_for(view.__name__)))
        return view(*args, **kwargs)

    return wrapped
