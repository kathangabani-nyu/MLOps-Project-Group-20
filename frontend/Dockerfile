FROM python:3.11-slim

RUN pip install --no-cache-dir streamlit pandas plotly

WORKDIR /app

COPY business_dashboard.py /app/

EXPOSE 5565

CMD ["streamlit", "run", "business_dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=5565", \
     "--server.enableCORS=false"]
