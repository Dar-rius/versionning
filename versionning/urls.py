from django.urls import path
from . import views

urlpatterns = [
    path("", views.load_csv_file, name="upload_csv"),
    path("train/", views.train_model, name="train_model"),
    path("visualisation/", views.visualisation, name="visualisation"),
]
