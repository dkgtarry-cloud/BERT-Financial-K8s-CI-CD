FROM python:3.14-slim

WORKDIR /app
COPY . .

RUN pip install  torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install  flask transformers joblib scikit-learn requests

EXPOSE 5000
CMD ["python", "app.py"]
