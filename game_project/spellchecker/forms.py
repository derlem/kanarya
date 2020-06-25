from django import forms


class QueryForm(forms.Form):
	sentence = forms.CharField(max_length=500)


class QueryFeedbackForm(forms.Form):
	#isHappy = forms.BooleanField(required = False)
	report = forms.CharField(max_length=1000)