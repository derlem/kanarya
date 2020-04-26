from django import forms
from .models import Activity, Report

#class DecisionForm(forms.Form):
class ActivityForm(forms.ModelForm):

	class Meta:
		model = Activity
		fields = ['name']

class ReportForm(forms.ModelForm):

	class Meta:
		model = Report
		fields = ['text']
