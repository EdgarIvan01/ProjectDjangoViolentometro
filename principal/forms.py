from django import forms

class Data(forms.Form):
    texto1 = forms.CharField(widget=forms.Textarea)
    texto2 = forms.CharField(widget=forms.Textarea)