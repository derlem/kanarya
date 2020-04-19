from django import forms
from .models import  Decision

#class DecisionForm(forms.Form):
class DecisionForm(forms.ModelForm):

	#btn = forms.CharField(max_length=100)

	class Meta:
		model = Decision
		fields = ['decision']