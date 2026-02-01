# ▶️ Как запустить локально

```python
# 1) создать/поднять PostgreSQL и .env
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2) подготовить БД
python -m db.db_init
python build_db.py

# 3) обучить модель
python train_mnist.py

# 4) стартовать приложение
streamlit run app.py
```
