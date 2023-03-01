from ray.job_submission import JobSubmissionClient
# If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
client = JobSubmissionClient("http://10.240.0.210:8265/")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python raylib_recommender.py",
    # Path to the local directory that contains the script.py file
    runtime_env={"working_dir": "./",
                 "pip": "requirements.txt"},
    #name you job
    submission_id="Christian_job_2"
)
print(job_id)