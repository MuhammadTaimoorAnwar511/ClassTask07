from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'pipeline_without_training',
    default_args=default_args,
    description='Automated pipeline without model training',
    schedule_interval=None,  # Run manually for now
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # Task 1: Generate Data
    generate_data = BashOperator(
        task_id='generate_data',
        bash_command='cd /opt/airflow/project && python LiveData.py'
    )

    # Task 2: Process Data
    process_data = BashOperator(
        task_id='process_data',
        bash_command='cd /opt/airflow/project && python EDA.py'
    )

    # Define task dependencies
    generate_data >> process_data
