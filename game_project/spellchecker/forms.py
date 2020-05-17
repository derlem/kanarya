from django import forms


class QueryForm(forms.Form):
	sentence = forms.CharField(max_length=500)