from django import forms
from .models import Activity

#class DecisionForm(forms.Form):
class ActivityForm(forms.ModelForm):

	class Meta:
		model = Activity
		fields = ['name']