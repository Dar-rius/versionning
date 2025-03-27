import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render, redirect
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from .models import *
from .forms import *
import os
import seaborn as sns 


def load_csv_file(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Supprime les fichiers CSV précédents
            CSV.objects.all().delete()
            TrainedModel.objects.all().delete()

            # Sauvegarde le nouveau fichier
            csv_instance = form.save()

            # Lecture du fichier pour validation
            try:
                df = pd.read_csv(csv_instance.file)

                # Validation des colonnes
                features = form.cleaned_data["features"].split(",")
                print(features)
                target = form.cleaned_data["target"]

                if not all(col in df.columns for col in features + [target]):
                    messages.error(
                        request,
                        "Les colonnes spécifiées n'existent pas dans le fichier CSV.",
                    )
                    csv_instance.delete()
                    return render(request, "upload_csv.html", {"form": form})

                messages.success(request, "Fichier CSV téléchargé avec succès!")
                return redirect("train_model")

            except Exception as e:
                messages.error(
                    request, f"Erreur lors du traitement du fichier: {str(e)}"
                )
                csv_instance.delete()
    else:
        form = CSVUploadForm()

    return render(request, "upload_csv.html", {"form": form})


def train_model(request):
    if request.method == "POST":
        form = ModelTrainingForm(request.POST)
        if form.is_valid():
            # Récupérer le dernier fichier CSV
            try:
                csv_file = CSV.objects.latest("created_at")

                # Lire le fichier
                df = pd.read_csv(csv_file.file)

                # Préparer les données
                features = csv_file.features.split(",")
                target = csv_file.target

                X = df[features]
                y = df[target]

                # Déterminer si c'est une classification ou régression
                is_classification = len(np.unique(y)) <= 10

                # Prétraitement
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split des données
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                # Sélection du modèle
                algorithm = form.cleaned_data["algorithm"]
                model = ""

                if is_classification:
                    if algorithm == "RandomForest":
                        model = RandomForestClassifier(random_state=42)
                    elif algorithm == "LogisticRegression":
                        model = LogisticRegression(random_state=42)
                    elif algorithm == "SVM":
                        model = SVC(random_state=42)
                else:
                    if algorithm == "RandomForest":
                        model = RandomForestRegressor(random_state=42)
                    elif algorithm == "LinearRegression":
                        model = LinearRegression()
                    elif algorithm == "SVM":
                        model = SVR()

                # Entraînement
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Sauvegarde des résultats
                trained_model = form.save(commit=False)

                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    trained_model.accuracy = accuracy
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    trained_model.mse = mse

                trained_model.save()

                return visualisation(
                    request, y_test, y_pred, is_classification, trained_model
                )

            except Exception as e:
                messages.error(request, f"Erreur lors de l'entraînement: {str(e)}")
    else:
        form = ModelTrainingForm()

    return render(request, "train_model.html", {"form": form})


def visualisation(request, y_test, y_pred, is_classification, trained_model):
    # Calculer les métriques détaillées
    metrics = {}
    plot_base64 = None
    residual_plot_base64 = None
    conf_matrix = None

    if is_classification:
        # Métriques de classification
        accuracy = accuracy_score(y_test, y_pred)
        metrics["Accuracy"] = f"{accuracy:.4f}"

        # Rapport de classification
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics["Classification Report"] = class_report

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Graphique de la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Convertir le graphique de la matrice de confusion
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

    else:
        # Métriques de régression
        mse = mean_squared_error(y_test, y_pred)
        metrics["Mean_Squared_Error"] = f"{mse:.4f}"

        # Scatter plot des valeurs actuelles vs prédites
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
            label="Perfect Prediction"
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.legend()
        
        # Convertir le scatter plot
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Plot des résidus
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        
        # Convertir le plot des résidus
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        residual_plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

    # Contexte pour le template
    context = {
        "metrics": metrics,
        "plot_base64": plot_base64,
        "residual_plot_base64": residual_plot_base64,
        "is_classification": is_classification,
        "trained_model": trained_model,
        "confusion_matrix": conf_matrix.tolist() if conf_matrix is not None else None,
    }
    
    return render(request, "model_results.html", context)
