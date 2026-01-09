from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    abort,
    send_from_directory,
    flash,
    session,
    make_response,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text
import re
import os
import json
import uuid
import base64
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import collections

app = Flask(__name__)


def _setup_audit_logger() -> logging.Logger:
    logger = logging.getLogger("smartdentalscan.audit")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    try:
        os.makedirs(app.instance_path, exist_ok=True)
        log_path = os.path.join(app.instance_path, "audit.log")
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    except Exception:
        # If file logging fails, fall back to default propagation.
        logger.propagate = True

    return logger


audit_logger = _setup_audit_logger()


def _hash_identifier(value: str) -> str:
    """One-way hash for correlating events without logging raw identifiers."""
    value = (value or "").strip().lower()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def audit_event(event: str, **fields) -> None:
    """Write an audit log line as JSON; avoid sensitive data."""
    try:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            "method": request.method if request else None,
            "path": request.path if request else None,
            "ip": request.headers.get("X-Forwarded-For", request.remote_addr) if request else None,
            "ua": (request.user_agent.string[:200] if request and request.user_agent else None),
        }
        for k, v in fields.items():
            if v is None:
                continue
            payload[k] = v
        audit_logger.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except Exception:
        pass

# Required for secure sessions (Flask-Login uses session cookies)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Limit upload size (5MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# -------- Database (SQLite) --------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------- Login (not enforced yet) --------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# CSRF protection for all modifying requests (POST/PUT/PATCH/DELETE)
csrf = CSRFProtect(app)


@app.errorhandler(413)
def request_entity_too_large(_e):
    # Keep response user-friendly. Rendering index avoids introducing a new page.
    return (
        render_template(
            "index.html",
            prediction=None,
            confidence=None,
            top3=None,
            image_uploaded=False,
            error="Upload too large. Max size is 5MB.",
        ),
        413,
    )


def hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = hash_password(password)

    def check_password(self, password: str) -> bool:
        return verify_password(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    image_path = db.Column(db.String(1024), nullable=False)
    predicted_label = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    top3_json = db.Column(db.Text, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("predictions", lazy=True))


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

# -------- Load class names --------
with open("classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------- Load model --------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

checkpoint = torch.load("model.pth", map_location="cpu")

if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# Remove "model." prefix
new_state_dict = collections.OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("model.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()

def ensure_prediction_notes_column() -> None:
    """Idempotent lightweight schema migration for SQLite.

    This project doesn't use Alembic; ensure legacy DBs get the new nullable column.
    """

    try:
        cols = db.session.execute(text("PRAGMA table_info(prediction)"))
        existing = {row[1] for row in cols}  # row[1] is column name
        if "notes" in existing:
            return
        db.session.execute(text("ALTER TABLE prediction ADD COLUMN notes TEXT"))
        db.session.commit()
    except Exception:
        db.session.rollback()


with app.app_context():
    db.create_all()
    ensure_prediction_notes_column()

# -------- Image transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def is_valid_email(email: str) -> bool:
    if not email:
        return False
    email = email.strip()
    # Simple, pragmatic validation (not full RFC)
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}


def is_allowed_upload(image_file) -> tuple[bool, str | None, str | None]:
    filename = secure_filename(image_file.filename or "")
    if not filename:
        return False, None, "Missing filename."

    _, ext = os.path.splitext(filename)
    ext = (ext or "").lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, ext, "Unsupported file type. Please upload a .jpg, .jpeg, or .png image."

    mimetype = (image_file.mimetype or "").lower()
    if mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return False, ext, "Unsupported MIME type. Please upload a JPEG or PNG image."

    return True, ext, None


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    success = None

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not is_valid_email(email):
            error = "Please enter a valid email address."
        elif len(password) < 8:
            error = "Password must be at least 8 characters long."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user is not None:
                error = "An account with that email already exists."
            else:
                user = User(email=email)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                success = "Account created. You can log in once login is enabled."

    return render_template("signup.html", error=error, success=success)


@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html", error=None)


@app.route("/login", methods=["POST"])
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    email_hash = _hash_identifier(email)

    if not is_valid_email(email):
        audit_event("login_attempt", outcome="invalid_email", email_hash=email_hash)
        return render_template("login.html", error="Please enter a valid email address.")

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        audit_event("login_attempt", outcome="failed", email_hash=email_hash)
        return render_template("login.html", error="Invalid email or password.")

    login_user(user)
    audit_event("login_attempt", outcome="success", user_id=user.id, email_hash=email_hash)
    return redirect(url_for("index"))


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/", methods=["GET"])
def index():
    prediction = None
    confidence = None
    top3 = None
    image_uploaded = False

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top3=top3,
        image_uploaded=image_uploaded
    )


@app.route("/predict", methods=["POST"])
def predict():
    prediction = None
    confidence = None
    top3 = None
    image_uploaded = False
    error = None

    # Guest mode: allow exactly one prediction per session.
    # Guests never persist uploads or DB rows.
    is_guest = not current_user.is_authenticated
    if is_guest and session.get("guest_prediction_used"):
        audit_event("prediction_attempt", outcome="blocked_guest_limit")
        flash("Guest scan already used. Please log in to scan again.", "warning")
        return redirect(url_for("login"))

    image_file = request.files.get("image")
    if image_file is None:
        audit_event("prediction_attempt", outcome="no_file", user_id=(current_user.id if current_user.is_authenticated else None))
        return render_template(
            "index.html",
            prediction=None,
            confidence=None,
            top3=None,
            image_uploaded=False,
            error="No file uploaded.",
        )

    allowed, validated_ext, upload_error = is_allowed_upload(image_file)
    if not allowed:
        audit_event(
            "prediction_attempt",
            outcome="rejected_upload",
            user_id=(current_user.id if current_user.is_authenticated else None),
            ext=validated_ext,
        )
        return render_template(
            "index.html",
            prediction=None,
            confidence=None,
            top3=None,
            image_uploaded=False,
            error=upload_error,
        )

    image_uploaded = True

    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Top-1
    conf, pred = torch.max(probs, 0)
    prediction = class_names[pred.item()]
    confidence = round(conf.item() * 100, 2)

    # Top-3
    top_probs, top_idxs = torch.topk(probs, 3)
    top3 = [
        (class_names[i], round(p.item() * 100, 2))
        for i, p in zip(top_idxs, top_probs)
    ]

    if is_guest:
        session["guest_prediction_used"] = True
        audit_event(
            "prediction_created",
            persisted=False,
            user_id=None,
            label=prediction,
            confidence=confidence,
        )
        flash("Guest mode: results are not saved. Log in to save history.", "info")
    else:
        # Persist results (upload + DB). Inference logic above remains unchanged.
        try:
            uploads_dir = os.path.join(app.root_path, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            # Ignore user-provided filenames; store a random name with validated extension.
            filename = f"{uuid.uuid4().hex}{validated_ext}"
            abs_path = os.path.join(uploads_dir, filename)

            # Stream was read by PIL; rewind before saving.
            image_file.stream.seek(0)
            image_file.save(abs_path)

            top3_payload = [{"label": label, "score": score} for label, score in top3]
            pred_row = Prediction(
                user_id=current_user.id,
                image_path=f"uploads/{filename}",
                predicted_label=prediction,
                confidence=confidence,
                top3_json=json.dumps(top3_payload),
            )
            db.session.add(pred_row)
            db.session.commit()
            audit_event(
                "prediction_created",
                persisted=True,
                user_id=current_user.id,
                prediction_id=pred_row.id,
                label=prediction,
                confidence=confidence,
            )
        except Exception:
            db.session.rollback()
            audit_event(
                "prediction_created",
                persisted=False,
                user_id=current_user.id,
                outcome="save_failed",
                label=prediction,
                confidence=confidence,
            )
            error = "Prediction computed, but saving failed. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top3=top3,
        image_uploaded=image_uploaded,
        error=error,
    )


@app.route("/uploads/<path:filename>", methods=["GET"])
@login_required
def uploaded_file(filename: str):
    # Prevent path traversal and only serve files owned by the current user.
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        abort(404)

    image_path = f"uploads/{safe_name}"
    owned = Prediction.query.filter_by(user_id=current_user.id, image_path=image_path).first()
    if owned is None:
        abort(404)

    uploads_dir = os.path.join(app.root_path, "uploads")
    abs_path = os.path.join(uploads_dir, safe_name)
    if not os.path.isfile(abs_path):
        abort(404)

    return send_from_directory(uploads_dir, safe_name)


@app.route("/history", methods=["GET"])
@login_required
def history():
    preds = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    items = []
    for p in preds:
        filename = (p.image_path or "").split("/")[-1]
        items.append(
            {
                "id": p.id,
                "label": p.predicted_label,
                "confidence": p.confidence,
                "created_at": p.created_at,
                "filename": filename,
            }
        )

    return render_template("history.html", items=items)


@app.route("/history/<int:prediction_id>", methods=["GET"])
@login_required
def history_detail(prediction_id: int):
    pred = db.session.get(Prediction, prediction_id)
    if pred is None:
        abort(404)
    if pred.user_id != current_user.id:
        abort(403)

    try:
        top3 = json.loads(pred.top3_json) if pred.top3_json else []
    except Exception:
        top3 = []

    filename = (pred.image_path or "").split("/")[-1]
    return render_template(
        "history_detail.html",
        prediction_id=pred.id,
        label=pred.predicted_label,
        confidence=pred.confidence,
        created_at=pred.created_at,
        filename=filename,
        top3=top3,
        notes=pred.notes,
    )


@app.route("/history/<int:prediction_id>/notes", methods=["POST"])
@login_required
def update_prediction_notes(prediction_id: int):
    pred = db.session.get(Prediction, prediction_id)
    if pred is None:
        abort(404)
    if pred.user_id != current_user.id:
        abort(403)

    notes = (request.form.get("notes") or "").strip()
    pred.notes = notes or None
    db.session.commit()

    flash("Notes saved.", "success")
    return redirect(url_for("history_detail", prediction_id=prediction_id))


@app.route("/history/<int:prediction_id>/report", methods=["GET"])
@login_required
def download_prediction_report(prediction_id: int):
    pred = db.session.get(Prediction, prediction_id)
    if pred is None:
        abort(404)
    if pred.user_id != current_user.id:
        abort(403)

    try:
        top3 = json.loads(pred.top3_json) if pred.top3_json else []
    except Exception:
        top3 = []

    # Embed the image as a data URI so the report is self-contained.
    data_uri = None
    try:
        filename = secure_filename((pred.image_path or "").split("/")[-1])
        if filename:
            uploads_dir = os.path.join(app.root_path, "uploads")
            abs_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(abs_path):
                _, ext = os.path.splitext(filename.lower())
                mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
                with open(abs_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                data_uri = f"data:{mime};base64,{encoded}"
    except Exception:
        data_uri = None

    html = render_template(
        "report.html",
        prediction_id=pred.id,
        label=pred.predicted_label,
        confidence=pred.confidence,
        created_at=pred.created_at,
        top3=top3,
        notes=pred.notes,
        image_data_uri=data_uri,
    )

    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename=SmartDentalScan_Report_{pred.id}.html"
    return resp


@app.route("/prediction/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_prediction(prediction_id: int):
    pred = db.session.get(Prediction, prediction_id)
    if pred is None:
        abort(404)
    if pred.user_id != current_user.id:
        abort(403)

    image_path = pred.image_path

    db.session.delete(pred)
    db.session.commit()

    audit_event("prediction_deleted", user_id=current_user.id, prediction_id=prediction_id)

    # Best-effort local file delete (record is already gone)
    try:
        if image_path:
            filename = secure_filename(image_path.split("/")[-1])
            uploads_dir = os.path.join(app.root_path, "uploads")
            abs_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(abs_path):
                os.remove(abs_path)
    except Exception:
        pass

    flash("Prediction deleted.")
    return redirect(url_for("history"))

if __name__ == "__main__":
    app.run(debug=True)
