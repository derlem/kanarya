from django import forms
from .models import Activity, Report

from users.models import Profile


#class DecisionForm(forms.Form):
class ActivityForm(forms.ModelForm):

	class Meta:
		model = Activity
		fields = ['name']

class ReportForm(forms.ModelForm):

	class Meta:
		model = Report
		fields = ['text']

"""
class ProfForm(forms.Form):

	answer = forms.CharField(max_length=5) # True or False
"""
