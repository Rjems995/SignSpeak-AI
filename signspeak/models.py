from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from . import db


def now():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=now, nullable=False)
    profile = db.relationship("UserProfile", uselist=False, backref="user", cascade="all, delete-orphan")
    settings = db.relationship("UserSettings", uselist=False, backref="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserProfile(db.Model):
    __tablename__ = "user_profiles"
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), primary_key=True)
    name = db.Column(db.String(80), nullable=False)


class UserSettings(db.Model):
    __tablename__ = "user_settings"
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), primary_key=True)
    values = db.Column(db.JSON, nullable=False, default=dict)


class TranslationHistory(db.Model):
    __tablename__ = "translation_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    text = db.Column(db.String(5000), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=now, nullable=False)

    def to_dict(self):
        stamp = self.created_at
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return {"id": self.id, "text": self.text, "created_at": stamp.isoformat()}


class PracticeSession(db.Model):
    __tablename__ = "practice_sessions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=now, nullable=False)
    results = db.relationship("PracticeResult", cascade="all, delete-orphan", backref="session")


class PracticeResult(db.Model):
    __tablename__ = "practice_results"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("practice_sessions.id"), nullable=False)
    expected = db.Column(db.String(1), nullable=False)
    predicted = db.Column(db.String(1), nullable=True)
    correct = db.Column(db.Boolean, nullable=False)


class LearningProgress(db.Model):
    __tablename__ = "learning_progress"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    letter = db.Column(db.String(1), nullable=False)
    completed_at = db.Column(db.DateTime(timezone=True), default=now, nullable=False)
    __table_args__ = (db.UniqueConstraint("user_id", "letter"),)

