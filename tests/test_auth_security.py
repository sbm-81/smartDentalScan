import json


def _create_user(app_module, email: str, password: str):
    with app_module.app.app_context():
        user = app_module.User(email=email.strip().lower())
        user.set_password(password)
        app_module.db.session.add(user)
        app_module.db.session.commit()
        app_module.db.session.refresh(user)
        return user


def _login(client, email: str, password: str):
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=False,
    )


def test_uploads_requires_login(client):
    resp = client.get("/uploads/any.png", follow_redirects=False)
    assert resp.status_code in (301, 302, 303)
    assert "/login" in (resp.headers.get("Location") or "")


def test_logged_in_user_cannot_access_other_users_image(app_module, client):
    user1 = _create_user(app_module, "u1@example.com", "password123")
    user2 = _create_user(app_module, "u2@example.com", "password123")

    # Create a prediction owned by user1.
    with app_module.app.app_context():
        pred = app_module.Prediction(
            user_id=user1.id,
            image_path="uploads/test.png",
            predicted_label="ClassA",
            confidence=99.0,
            top3_json=json.dumps(
                [
                    {"label": "ClassA", "score": 99.0},
                    {"label": "ClassB", "score": 1.0},
                    {"label": "ClassC", "score": 0.0},
                ]
            ),
        )
        app_module.db.session.add(pred)
        app_module.db.session.commit()

    # Log in as user2 and attempt to access user1's file.
    login_resp = _login(client, "u2@example.com", "password123")
    assert login_resp.status_code in (301, 302, 303)

    resp = client.get("/uploads/test.png", follow_redirects=False)
    # Route checks ownership first and should 404 for non-owners.
    assert resp.status_code == 404


def test_history_redirects_when_not_authenticated(client):
    resp = client.get("/history", follow_redirects=False)
    assert resp.status_code in (301, 302, 303)
    assert "/login" in (resp.headers.get("Location") or "")
