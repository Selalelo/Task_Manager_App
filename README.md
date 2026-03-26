# Intelligent Task Management System with Machine Learning

An AI-powered task management platform designed for enterprise environments, enabling intelligent task allocation and data-driven HR decision-making.

This system uses machine learning models to:
- Automatically assign tasks to the most suitable employees
- Predict salaries for new hires based on market data

---

## Project Overview

This project simulates a real-world company environment with:
- Employees having different skill sets and workloads
- A Project Manager assigning tasks
- An Admin managing users
- An HR module for salary estimation

The system leverages machine learning to improve task allocation efficiency, workforce productivity, and hiring decisions.

---

## Key Features

### Intelligent Task Assignment
Automatically assigns tasks using a Random Forest model, considering:
- Employee experience
- Tech stack
- Current workload (number of tasks)
- Task type

> Ensures the best-fit employee is selected for each task.

### HR Salary Prediction System
Predicts salary for new employees using ML, based on:
- Location
- Experience
- Tech stack
- External data from the Glassdoor API

> Helps HR make data-driven compensation decisions.

### Role-Based System

| Role | Responsibilities |
|------|-----------------|
| Admin | Add/manage employees and system users |
| Project Manager | Create and assign tasks (via ML model) |
| Employees | Receive and complete assigned tasks |
| HR | Estimate salaries for new hires |

---

##  System Architecture

```
Users (Admin / PM / HR / Employee)
            │
            ▼
        Backend API
            │
     ┌───────────────┬────────────────┐
     ▼               ▼                ▼
 Task Assignment   Salary Model     Database
 (Random Forest)   (Regression ML)  (Employees, Tasks)
     │               │
     ▼               ▼
 Task Allocation   Salary Prediction
```

---

##  Machine Learning Models

###  Task Assignment Model
- **Algorithm:** Random Forest Classifier
- **Goal:** Assign the most suitable employee to a task
- **Features:** Experience level, tech stack match, current workload, task type

###  Salary Prediction Model
- **Algorithm:** Regression-based model (trained on external data)
- **Data Source:** Glassdoor API
- **Features:** Location, experience, tech stack

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn (Random Forest) |
| Database | PostgreSQL / SQL |
| External APIs | Glassdoor API |

---

##  Workflow

### Task Assignment Flow
```
Project Manager creates task
        │
        ▼
ML Model evaluates employees
        │
        ▼
Best candidate selected
        │
        ▼
Task assigned automatically
```

### Salary Prediction Flow
```
HR inputs candidate details
        │
        ▼
Fetch market data (Glassdoor API)
        │
        ▼
ML model predicts salary
        │
        ▼
HR decision support
```

---

##  Project Structure

```
task-management-ml/
│
├── models/
│   ├── task_assignment_model.pkl
│   └── salary_prediction_model.pkl
│
├── api/
│   ├── routes.py
│   └── controllers/
│
├── data/
│   └── training_data.csv
│
├── services/
│   ├── ml_service.py
│   └── glassdoor_api.py
│
├── database/
│   └── schema.sql
│
├── main.py
├── requirements.txt
└── README.md
```

---

##  How to Run

```bash
# Clone repository
git clone https://github.com/your-username/task-management-ml.git
cd task-management-ml

# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn main:app --reload
```

---

## Skills Demonstrated

- Machine Learning (Random Forest, Regression)
- Backend API development
- System design & architecture
- Data-driven decision making
- Integration with external APIs
- Role-based access systems

---

## Team Contribution

This was a group project. Responsibilities included:
- Designing system architecture
- Implementing machine learning models
- Developing backend APIs
- Integrating external salary data APIs

---

## Future Improvements

- Add real-time task tracking dashboard
- Improve model accuracy with larger datasets
- Add authentication & authorization layers
- Deploy system on cloud infrastructure

---

## 📌 Summary

This project demonstrates how machine learning can be applied in enterprise systems to automate decision-making, improve operational efficiency, and support HR analytics.
