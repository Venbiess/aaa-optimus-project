from locust import HttpUser, task, between


class ModelUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_model(self):
        with open("locust/test_image.jpg", "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            self.client.post("/upload", files=files)
