#RUN echo "Start build"
FROM python:3.8
RUN python -m pip install --upgrade pip \
    && pip install click==8.1.2 \
    && pip install Flask==2.1.1 \
    && pip install Flask-Cors==3.0.10 \
    && pip install Flask-SQLAlchemy==2.5.1 \
    && pip install greenlet==1.1.2 \
    && pip install gunicorn==20.1.0 \
    && pip install importlib-metadata==4.11.3 \
    && pip install itsdangerous==2.1.2 \
    && pip install Jinja2==3.1.1 \
    && pip install MarkupSafe==2.1.1 \
    && pip install psycopg2-binary==2.9.3 \
    && pip install six==1.16.0 \
    && pip install SQLAlchemy==1.4.34 \
    && pip install Werkzeug==2.1.0 \
    && pip install zipp==3.7.0

copy app.py /src/app.py
ENTRYPOINT python3 app.py