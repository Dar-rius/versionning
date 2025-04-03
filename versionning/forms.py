from django import forms
from .models import CSV, TrainedModel
import pandas as pd


class CSVUploadForm(forms.ModelForm):
    features = forms.CharField(
        widget=forms.Textarea(
            attrs={"rows": 3, "placeholder": "Comma-separated feature columns"}
        ),
        help_text="Enter feature column names separated by commas",
    )
    target = forms.CharField(max_length=255, help_text="Enter the target column name")

    class Meta:
        model = CSV
        fields = ["file", "features", "target"]

    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get("file")

        if file:
            try:
                # Lire le fichier CSV pour validation
                df = pd.read_csv(file)
                # Réinitialiser le curseur pour permettre une lecture ultérieure
                file.seek(0)

                # Vérifier que les colonnes spécifiées existent dans le CSV
                features = cleaned_data.get("features", "").split(",")
                target = cleaned_data.get("target", "")

                if not all(col.strip() in df.columns for col in features + [target]):
                    raise forms.ValidationError(
                        "Les colonnes spécifiées n'existent pas dans le fichier."
                    )
            except Exception as e:
                raise forms.ValidationError(
                    f"Erreur de lecture du fichier CSV : {str(e)}"
                )

        return cleaned_data



class ModelTrainingForm(forms.ModelForm):
    class Meta:
        model = TrainedModel
        fields = ["algorithm", "csv_file"]
