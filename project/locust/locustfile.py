# from locust import HttpUser, task, between
#
#
#
# class ModelUser(HttpUser):
#     wait_time = between(3, 5)
#
#     @task
#     def test_model(self):
#         with open("locust/test_image.jpg", "rb") as f:
#             file = {"file": ("test_image.jpg", f, "image/jpeg")}
#             data = {
#                 "model": "unet",
#                 "format": "jpeg",
#                 "blur_level": "47"
#             }
#             self.client.post("http://localhost:8000/upload/", files=file, data=data)
#
