import random
from locust import HttpLocust, TaskSet, task


class UserBehavior(TaskSet):
    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        self.url = "/predictions/fizbuz_package"
        self.headers = {"Content-Type": "application/json"}

    @task(10)
    def success(self):
        data = {'input.1': random.randint(0, 1000)}
        self.client.post(self.url, headers=self.headers, json=data)

    @task(1)
    def failure_empty_body(self):
        data = {}
        self.client.post(self.url, headers=self.headers, json=data)

    @task(1)
    def failure_wrong_name(self):
        data = {'chaos': 143}
        self.client.post(self.url, headers=self.headers, json=data)


class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    host = "http://localhost:8080"
