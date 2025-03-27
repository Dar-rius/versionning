from django.db import models
import numpy as np
import os
from django.db.models.signals import post_delete
from django.dispatch import receiver


class CSV(models.Model):
    file = models.FileField(upload_to="csv_files/")
    filename = models.CharField(max_length=255)
    features = models.TextField()  # Stockera la liste des colonnes de features
    target = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Automatically set the filename if not already set
        if not self.filename:
            # Get the original filename from the file field
            self.filename = os.path.basename(self.file.name)

        # Call the parent class's save method
        super().save(*args, **kwargs)

    def __str__(self):
        return self.filename

@receiver(post_delete, sender=CSV)
def delete_file_on_delete(sender, instance, **kwargs):
    # Check if file exists and delete it
    if instance.file:
        try:
            if os.path.isfile(instance.file.path):
                os.remove(instance.file.path)
        except Exception as e:
            print(f"Error deleting file: {e}")

class TrainedModel(models.Model):
    ALGORITHM_CHOICES = [
        ("RandomForest", "Random Forest"),
        ("LogisticRegression", "Logistic Regression"),
        ("LinearRegression", "Linear Regression"),
        ("SVM", "Support Vector Machine"),
    ]

    csv_file = models.ForeignKey(CSV, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES)
    accuracy = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} - {self.csv_file.filename}"
