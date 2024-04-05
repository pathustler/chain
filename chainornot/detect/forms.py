from django import forms

class SingleInputFieldForm(forms.Form):
  text_field = forms.CharField(label=False, max_length=100)