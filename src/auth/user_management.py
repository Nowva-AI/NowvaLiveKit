"""
User Account Management
Handles user registration, account creation, and password management
"""

import secrets
import string
from db.database import SessionLocal
from db.models import User
from utils.username_generator import generate_username
from auth.security import hash_password


def generate_temporary_password(length: int = 16) -> str:
    """
    Generate a secure temporary password.
    This will be replaced when user sets their password via email.

    Args:
        length: Length of the password (default 16)

    Returns:
        Randomly generated password containing letters, digits, and punctuation

    Note:
        This generates a cryptographically secure random password suitable
        for temporary use until the user completes password setup via email.
    """
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def create_user_account(first_name: str, email: str) -> tuple[User, str]:
    """
    Create a new user account in the database.

    Args:
        first_name: User's first name
        email: User's email address

    Returns:
        Tuple of (User object, generated username)

    Raises:
        Exception if user creation fails

    Note:
        - Generates a unique username from the first name
        - Creates a temporary password (to be replaced via email verification)
        - If email already exists, returns the existing user instead of creating a duplicate
        - TODO: Hash password properly when password reset is implemented (use bcrypt/passlib)
    """
    db = SessionLocal()
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            print("[WARNING] A user with that email already exists — returning it")
            return existing_user, existing_user.username

        # Generate unique username
        username = generate_username(first_name, db)

        # Generate temporary password (to be replaced via email)
        temp_password = generate_temporary_password()

        # Create new user
        new_user = User(
            username=username,
            name=first_name,
            email=email,
            password_hash=hash_password(temp_password)
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        print(f"[SUCCESS] Created user account: {username}")
        return new_user, username

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to create user: {str(e)}")
        raise
    finally:
        db.close()
