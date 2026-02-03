# SmartGridSecurity
Web application for Anomaly Detection in Smart Grid Systems.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed on your system:

- **PyCharm** - [Download here](https://www.jetbrains.com/pycharm/)
- **Python 3.7.9 (64-bit)** - [Download here](https://www.python.org/downloads/release/python-379/)
- **Git** - [Download here](https://git-scm.com/downloads)

## Installation Steps

### 1. Clone the Repository

Open terminal and run:

```bash
git clone https://github.com/x1tedbtw/SmartGridSecurityApp
cd SmartGridSecurity
```

### 2. Set Python Interpreter
- Select Python 3.7.9


### 3. Create Virtual Environment

```bash
python -m venv venv
```

This creates a `venv` folder in your project directory.

### 4. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 7. Set Up Database

```bash
python manage.py migrate
```

This creates the `db.sqlite3` database file and sets up all necessary tables.

### 9. Run the Application

```bash
python manage.py runserver
```

### 10. Access the Application

Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
