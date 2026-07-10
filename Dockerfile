# WebHarbor — slim, self-contained image.
# 17 Flask mirror sites + control plane on :8101.

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8

RUN pip3 install --no-cache-dir \
    Flask==3.1.0 \
    Flask-SQLAlchemy==3.1.1 \
    Flask-Login==0.6.3 \
    Flask-WTF==1.2.2 \
    Flask-Bcrypt==1.0.1 \
    bcrypt==4.2.1 \
    Werkzeug==3.1.3 \
    Jinja2==3.1.4 \
    SQLAlchemy==2.0.36 \
    WTForms==3.2.1 \
    email-validator==2.2.0 \
    Pillow==11.0.0 \
    requests==2.32.3 \
    blinker==1.9.0 \
    certifi==2026.6.17 \
    charset-normalizer==3.4.9 \
    click==8.4.2 \
    dnspython==2.8.0 \
    greenlet==3.5.3 \
    idna==3.18 \
    itsdangerous==2.2.0 \
    MarkupSafe==3.0.3 \
    typing-extensions==4.16.0 \
    urllib3==2.7.0

WORKDIR /opt/WebSyn

# Sites tree. Build context must contain the heavy assets (instance_seed/,
# static/images/, static/external_cache/) — either commit them locally or
# run scripts/fetch_assets.sh to pull them from Hugging Face first.
COPY sites/ /opt/WebSyn/

COPY websyn_start.sh    /opt/websyn_start.sh
COPY control_server.py  /opt/control_server.py
COPY site_runner.py     /opt/site_runner.py
RUN chmod +x /opt/websyn_start.sh

EXPOSE 8101 40000-40016

CMD ["/opt/websyn_start.sh"]
